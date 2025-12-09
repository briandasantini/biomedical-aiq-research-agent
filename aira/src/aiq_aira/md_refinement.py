"""
MD Refinement integration for Biomedical AI-Q Research Agent.

This module integrates the GROMACS MD pipeline with the virtual screening
workflow, adding pose refinement and free energy calculations after DiffDock.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from langgraph.types import StreamWriter

logger = logging.getLogger(__name__)

# Try to import the MD pipeline - it should be installed as a dependency
try:
    from gromacs_md import MDPipeline, MDConfig, MDResult
    MD_AVAILABLE = True
except ImportError:
    MD_AVAILABLE = False
    logger.warning("gromacs_md package not installed. MD refinement will be skipped.")

# Check for gmx_MMPBSA availability
GMX_MMPBSA_PATH = Path("/opt/conda/envs/mmpbsa/bin/gmx_MMPBSA")
MMPBSA_AVAILABLE = GMX_MMPBSA_PATH.exists()


@dataclass
class MDRefinementResult:
    """Results from MD refinement of docked poses."""
    success: bool
    ligand_id: str
    pose_id: int
    binding_energy: Optional[float] = None  # kcal/mol
    binding_energy_std: Optional[float] = None
    ligand_rmsd: Optional[float] = None  # Angstrom
    pose_stable: bool = False
    refined_structure_path: Optional[str] = None
    trajectory_path: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_report_string(self) -> str:
        """Format result for inclusion in report."""
        if not self.success:
            return f"- Ligand {self.ligand_id} Pose {self.pose_id}: MD refinement failed ({self.error_message})"
        
        stability = "stable" if self.pose_stable else "unstable"
        energy_str = f"{self.binding_energy:.1f} ± {self.binding_energy_std:.1f}" if self.binding_energy else "N/A"
        
        return (
            f"- **Ligand {self.ligand_id} Pose {self.pose_id}**: "
            f"ΔG = {energy_str} kcal/mol, "
            f"RMSD = {self.ligand_rmsd:.2f} Å ({stability})"
        )


class MDRefinementIntegration:
    """
    Integrates GROMACS MD refinement into the virtual screening workflow.
    
    This class handles:
    1. Converting DiffDock output (.mol) to PDB format
    2. Combining protein + ligand into complex
    3. Running MD refinement pipeline
    4. Collecting and formatting results for the report
    """
    
    def __init__(
        self,
        output_dir: str,
        simulation_time_ns: float = 5.0,
        use_docker: bool = True,
        skip_mmpbsa: bool = False,
        gpu_id: int = 0,
    ):
        """
        Initialize MD refinement integration.
        
        Args:
            output_dir: Directory for MD output files
            simulation_time_ns: Simulation time in nanoseconds
            use_docker: Use Docker for GROMACS execution
            skip_mmpbsa: Skip MM-PBSA calculation (faster)
            gpu_id: GPU device ID
        """
        # Convert to absolute path for Docker volume mounts
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.simulation_time_ns = simulation_time_ns
        self.use_docker = use_docker
        self.skip_mmpbsa = skip_mmpbsa
        self.gpu_id = gpu_id
        
        if not MD_AVAILABLE:
            logger.warning("MD pipeline not available - refinement will be skipped")
    
    async def refine_docked_poses(
        self,
        protein_pdb_path: str,
        docked_ligands_dir: str,
        writer: StreamWriter,
        max_poses: int = 3,
        ligand_name: str = "LIG",
    ) -> list[MDRefinementResult]:
        """
        Run MD refinement on docked poses from DiffDock.
        
        Args:
            protein_pdb_path: Path to protein PDB file
            docked_ligands_dir: Directory containing .mol files from DiffDock
            writer: StreamWriter for progress updates
            max_poses: Maximum number of poses to refine
            ligand_name: Residue name for ligand
            
        Returns:
            List of MDRefinementResult objects
        """
        if not MD_AVAILABLE:
            writer({"md_refinement": "\n MD refinement skipped - gromacs_md package not installed \n"})
            return []
        
        writer({"md_refinement": "\n Starting MD refinement of docked poses \n"})
        
        # Find all .mol files from DiffDock
        ligands_dir = Path(docked_ligands_dir)
        mol_files = sorted(ligands_dir.glob("ligand_*.mol"))[:max_poses]
        
        if not mol_files:
            writer({"md_refinement": "\n No .mol files found from DiffDock \n"})
            return []
        
        writer({"md_refinement": f"\n Found {len(mol_files)} poses to refine \n"})
        
        results = []
        
        for mol_file in mol_files:
            # Parse ligand ID and pose ID from filename (e.g., ligand_0_1.mol)
            parts = mol_file.stem.split("_")
            ligand_id = parts[1] if len(parts) > 1 else "0"
            pose_id = int(parts[2]) if len(parts) > 2 else 0
            
            writer({"md_refinement": f"\n Refining ligand {ligand_id} pose {pose_id}... \n"})
            
            try:
                result = await self._refine_single_pose(
                    protein_pdb_path=protein_pdb_path,
                    ligand_mol_path=str(mol_file),
                    ligand_id=ligand_id,
                    pose_id=pose_id,
                    ligand_name=ligand_name,
                    writer=writer,
                )
                results.append(result)
                
                if result.success:
                    writer({"md_refinement": f"\n ✓ Ligand {ligand_id} pose {pose_id}: "
                           f"ΔG={result.binding_energy:.1f} kcal/mol, "
                           f"RMSD={result.ligand_rmsd:.2f} Å \n"})
                else:
                    writer({"md_refinement": f"\n ✗ Ligand {ligand_id} pose {pose_id}: {result.error_message} \n"})
                    
            except Exception as e:
                logger.error(f"MD refinement failed for {mol_file}: {e}")
                results.append(MDRefinementResult(
                    success=False,
                    ligand_id=ligand_id,
                    pose_id=pose_id,
                    error_message=str(e),
                ))
        
        return results
    
    async def _refine_single_pose(
        self,
        protein_pdb_path: str,
        ligand_mol_path: str,
        ligand_id: str,
        pose_id: int,
        ligand_name: str,
        writer: StreamWriter,
    ) -> MDRefinementResult:
        """Refine a single docked pose."""
        
        # Create working directory for this pose
        pose_dir = self.output_dir / f"ligand_{ligand_id}_pose_{pose_id}"
        pose_dir.mkdir(exist_ok=True)
        
        try:
            # Step 1: Convert .mol to .pdb
            ligand_pdb = await self._convert_mol_to_pdb(ligand_mol_path, pose_dir)
            
            if not ligand_pdb:
                return MDRefinementResult(
                    success=False,
                    ligand_id=ligand_id,
                    pose_id=pose_id,
                    error_message="Failed to convert MOL to PDB",
                )
            
            # Step 2: Create complex PDB (protein + ligand)
            complex_pdb = self._create_complex_pdb(
                protein_pdb_path, ligand_pdb, pose_dir, ligand_name
            )
            
            # Step 3: Run MD pipeline
            config = MDConfig(
                simulation_time_ns=self.simulation_time_ns,
                gpu_id=self.gpu_id,
                equilibration_time_ps=100.0,  # Shorter for screening
            )
            
            # Ensure absolute path for Docker volume mounts
            md_work_dir = (pose_dir / "md_run").resolve()
            
            pipeline = MDPipeline(
                work_dir=md_work_dir,
                config=config,
                use_docker=self.use_docker,
            )
            
            # Run in executor to not block event loop
            # Ensure absolute path for Docker
            complex_pdb_abs = str(Path(complex_pdb).resolve())
            
            loop = asyncio.get_event_loop()
            md_result = await loop.run_in_executor(
                None,
                lambda: pipeline.run(
                    complex_pdb=complex_pdb_abs,
                    ligand_name=ligand_name,
                    skip_mmpbsa=self.skip_mmpbsa,
                )
            )
            
            if md_result.success:
                # Run MM-GBSA if not skipped and available
                binding_energy = md_result.binding_free_energy
                binding_energy_std = md_result.binding_energy_std
                
                if not self.skip_mmpbsa and MMPBSA_AVAILABLE:
                    writer({"md_refinement": f"\n   Running MM-GBSA for ligand {ligand_id} pose {pose_id}... \n"})
                    mmpbsa_result = await self._run_mmpbsa(
                        sim_dir=pose_dir / "md_run" / "03_simulation",
                        output_dir=pose_dir / "md_run" / "05_mmpbsa",
                        ligand_name=ligand_name,
                    )
                    if mmpbsa_result.get("binding_energy") is not None:
                        binding_energy = mmpbsa_result["binding_energy"]
                        binding_energy_std = mmpbsa_result.get("binding_energy_std", 0.0)
                
                return MDRefinementResult(
                    success=True,
                    ligand_id=ligand_id,
                    pose_id=pose_id,
                    binding_energy=binding_energy,
                    binding_energy_std=binding_energy_std,
                    ligand_rmsd=md_result.average_rmsd,
                    pose_stable=md_result.pose_stable,
                    refined_structure_path=str(md_result.refined_structure),
                    trajectory_path=str(md_result.trajectory),
                )
            else:
                return MDRefinementResult(
                    success=False,
                    ligand_id=ligand_id,
                    pose_id=pose_id,
                    error_message=md_result.error_message,
                )
                
        except Exception as e:
            logger.error(f"MD refinement error: {e}")
            return MDRefinementResult(
                success=False,
                ligand_id=ligand_id,
                pose_id=pose_id,
                error_message=str(e),
            )
    
    async def _run_mmpbsa(
        self,
        sim_dir: Path,
        output_dir: Path,
        ligand_name: str = "LIG",
    ) -> dict:
        """Run MM-GBSA calculation using gmx_MMPBSA."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tpr_file = sim_dir / "md.tpr"
        xtc_file = sim_dir / "md.xtc"
        top_file = sim_dir / "topol.top"
        
        if not all(f.exists() for f in [tpr_file, xtc_file, top_file]):
            logger.warning("Missing simulation files for MM-GBSA")
            return {}
        
        # Create index file using GROMACS Docker
        index_cmd = f"""docker run --rm -v {sim_dir}:/data -w /data nvcr.io/hpc/gromacs:2023.2 \
            bash -c "echo -e 'q' | gmx make_ndx -f md.tpr -o index.ndx 2>/dev/null" """
        subprocess.run(index_cmd, shell=True, capture_output=True)
        
        index_file = sim_dir / "index.ndx"
        if not index_file.exists():
            logger.warning("Failed to create index file for MM-GBSA")
            return {}
        
        shutil.copy(index_file, output_dir / "index.ndx")
        
        # Create MM-GBSA input file
        mmpbsa_input = output_dir / "mmpbsa.in"
        with open(mmpbsa_input, "w") as f:
            f.write(f"""# MM-GBSA input file
&general
    sys_name="Protein_{ligand_name}",
    startframe=1,
    endframe=100,
    interval=1,
    verbose=2,
    forcefields="oldff/leaprc.ff99SB,leaprc.gaff2"
/
&gb
    igb=5,
    saltcon=0.150,
/
""")
        
        # Setup environment for gmx_MMPBSA
        env = os.environ.copy()
        conda_bin = Path("/opt/conda/envs/mmpbsa/bin")
        env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
        env["AMBERHOME"] = "/opt/conda/envs/mmpbsa"
        
        # Run gmx_MMPBSA
        cmd = [
            str(GMX_MMPBSA_PATH),
            "-O",
            "-i", str(mmpbsa_input),
            "-cs", str(tpr_file),
            "-ci", str(output_dir / "index.ndx"),
            "-cg", "1", "13",  # Protein=1, Ligand=13 (typical)
            "-ct", str(xtc_file),
            "-cp", str(top_file),
            "-o", str(output_dir / "FINAL_RESULTS_MMPBSA.dat"),
            "-eo", str(output_dir / "FINAL_RESULTS_MMPBSA.csv"),
        ]
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=output_dir,
                    env=env,
                    timeout=1800,  # 30 min timeout
                )
            )
            
            # Parse results
            results_file = output_dir / "FINAL_RESULTS_MMPBSA.dat"
            if results_file.exists():
                return self._parse_mmpbsa_results(results_file)
                
        except subprocess.TimeoutExpired:
            logger.warning("MM-GBSA calculation timed out")
        except Exception as e:
            logger.warning(f"MM-GBSA calculation failed: {e}")
        
        return {}
    
    def _parse_mmpbsa_results(self, results_file: Path) -> dict:
        """Parse gmx_MMPBSA output file."""
        results = {
            "binding_energy": None,
            "binding_energy_std": None,
            "components": {}
        }
        
        with open(results_file) as f:
            content = f.read()
        
        in_delta = False
        for line in content.split("\n"):
            if "Delta (Complex - Receptor - Ligand):" in line:
                in_delta = True
                continue
            
            if in_delta:
                parts = line.split()
                if len(parts) >= 3:
                    if parts[0] == "ΔTOTAL":
                        try:
                            results["binding_energy"] = float(parts[1])
                            results["binding_energy_std"] = float(parts[3]) if len(parts) > 3 else 0.0
                        except (ValueError, IndexError):
                            pass
                    elif parts[0].startswith("Δ"):
                        try:
                            results["components"][parts[0]] = float(parts[1])
                        except (ValueError, IndexError):
                            pass
        
        if results["binding_energy"] is not None:
            logger.info(f"MM-GBSA: ΔG = {results['binding_energy']:.2f} ± {results['binding_energy_std']:.2f} kcal/mol")
        
        return results
    
    async def _convert_mol_to_pdb(self, mol_path: str, output_dir: Path) -> Optional[Path]:
        """Convert MOL file to PDB format using Open Babel."""
        
        output_pdb = output_dir / "ligand.pdb"
        
        # Try using obabel (Open Babel)
        try:
            import subprocess
            
            # First try docker
            if self.use_docker:
                cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{Path(mol_path).parent}:/input",
                    "-v", f"{output_dir}:/output",
                    "obabel/obabel:latest",
                    "-imol", f"/input/{Path(mol_path).name}",
                    "-opdb", "-O", f"/output/ligand.pdb",
                    "--gen3d"  # Generate 3D coordinates if needed
                ]
            else:
                cmd = [
                    "obabel",
                    "-imol", mol_path,
                    "-opdb", "-O", str(output_pdb),
                    "--gen3d"
                ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and output_pdb.exists():
                # Add residue name to ligand atoms
                self._fix_ligand_residue_name(output_pdb, "LIG")
                return output_pdb
            else:
                logger.warning(f"obabel conversion failed: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"obabel not available or failed: {e}")
        
        # Fallback: try RDKit
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            mol = Chem.MolFromMolFile(mol_path)
            if mol is None:
                return None
            
            # Add hydrogens if needed
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            
            Chem.MolToPDBFile(mol, str(output_pdb))
            self._fix_ligand_residue_name(output_pdb, "LIG")
            return output_pdb
            
        except ImportError:
            logger.warning("RDKit not available for MOL to PDB conversion")
        except Exception as e:
            logger.error(f"RDKit conversion failed: {e}")
        
        return None
    
    def _fix_ligand_residue_name(self, pdb_path: Path, res_name: str = "LIG") -> None:
        """Update residue name in ligand PDB to specified name."""
        
        with open(pdb_path, "r") as f:
            lines = f.readlines()
        
        with open(pdb_path, "w") as f:
            for line in lines:
                if line.startswith(("ATOM", "HETATM")):
                    # Replace residue name (columns 17-20)
                    line = line[:17] + f"{res_name:>3}" + line[20:]
                f.write(line)
    
    def _create_complex_pdb(
        self,
        protein_pdb: str,
        ligand_pdb: Path,
        output_dir: Path,
        ligand_name: str = "LIG",
    ) -> Path:
        """Combine protein and ligand into a single PDB file."""
        
        complex_pdb = output_dir / "complex.pdb"
        
        with open(complex_pdb, "w") as out:
            # Write protein atoms
            with open(protein_pdb, "r") as prot:
                for line in prot:
                    if line.startswith(("ATOM", "HETATM")):
                        # Skip water and existing ligands
                        res_name = line[17:20].strip()
                        if res_name not in ["HOH", "WAT", "SOL"]:
                            out.write(line)
                    elif line.startswith("TER"):
                        out.write(line)
            
            # Add TER between protein and ligand
            out.write("TER\n")
            
            # Write ligand atoms as HETATM
            with open(ligand_pdb, "r") as lig:
                for line in lig:
                    if line.startswith("ATOM"):
                        # Convert ATOM to HETATM for ligand
                        out.write("HETATM" + line[6:])
                    elif line.startswith("HETATM"):
                        out.write(line)
            
            out.write("TER\n")
            out.write("END\n")
        
        return complex_pdb


def format_md_results_for_report(results: list[MDRefinementResult]) -> str:
    """
    Format MD refinement results for inclusion in the research report.
    
    Args:
        results: List of MDRefinementResult objects
        
    Returns:
        Formatted markdown string for the report
    """
    if not results:
        return "\n*MD refinement was not performed.*\n"
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    output = []
    output.append("\n### Molecular Dynamics Refinement Results\n")
    output.append(f"MD simulations were performed on {len(results)} docked poses ")
    output.append(f"to assess binding stability and calculate binding free energies.\n")
    
    if successful:
        # Sort by binding energy (most negative first)
        successful.sort(key=lambda x: x.binding_energy if x.binding_energy else 0)
        
        output.append("\n**Refined Poses (ranked by binding energy):**\n")
        for result in successful:
            output.append(result.to_report_string() + "\n")
        
        # Highlight best pose
        best = successful[0]
        output.append(f"\n**Best candidate:** Ligand {best.ligand_id} Pose {best.pose_id} ")
        output.append(f"with predicted binding free energy of {best.binding_energy:.1f} kcal/mol ")
        if best.pose_stable:
            output.append("and stable binding pose during MD simulation.\n")
        else:
            output.append("(note: pose showed some instability during simulation).\n")
    
    if failed:
        output.append(f"\n*{len(failed)} poses failed MD refinement and were excluded.*\n")
    
    return "".join(output)


async def run_md_refinement(
    protein_pdb_path: str,
    docked_ligands_dir: str,
    output_dir: str,
    writer: StreamWriter,
    simulation_time_ns: float = 5.0,
    max_poses: int = 3,
    skip_mmpbsa: bool = False,
) -> tuple[list[MDRefinementResult], str]:
    """
    Convenience function to run MD refinement on DiffDock results.
    
    Args:
        protein_pdb_path: Path to protein PDB
        docked_ligands_dir: Directory with DiffDock .mol outputs
        output_dir: Output directory for MD results
        writer: StreamWriter for progress updates
        simulation_time_ns: Simulation time
        max_poses: Maximum poses to refine
        skip_mmpbsa: Skip free energy calculation
        
    Returns:
        Tuple of (results list, formatted report string)
    """
    integration = MDRefinementIntegration(
        output_dir=output_dir,
        simulation_time_ns=simulation_time_ns,
        skip_mmpbsa=skip_mmpbsa,
    )
    
    results = await integration.refine_docked_poses(
        protein_pdb_path=protein_pdb_path,
        docked_ligands_dir=docked_ligands_dir,
        writer=writer,
        max_poses=max_poses,
    )
    
    report_text = format_md_results_for_report(results)
    
    return results, report_text

