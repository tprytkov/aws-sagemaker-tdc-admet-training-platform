from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GPU_TORCH_INDEX = "--extra-index-url https://download.pytorch.org/whl/cu124"
CPU_TORCH_INDEX = "--extra-index-url https://download.pytorch.org/whl/cpu"


def _pip_dependencies(path: Path) -> list[str]:
    environment = yaml.safe_load(path.read_text(encoding="utf-8"))
    pip_sections = [item["pip"] for item in environment["dependencies"] if isinstance(item, dict)]
    assert len(pip_sections) == 1
    return pip_sections[0]


def test_gpu_environment_pins_verified_cuda_12_4_pytorch() -> None:
    dependencies = _pip_dependencies(ROOT / "environment-chemprop-gpu.yml")

    assert GPU_TORCH_INDEX in dependencies
    assert "torch==2.6.0+cu124" in dependencies
    assert "-r requirements-chemprop.txt" in dependencies
    assert all("2.13.0" not in dependency for dependency in dependencies)


def test_local_environment_is_explicitly_cpu_only_and_separate() -> None:
    gpu = _pip_dependencies(ROOT / "environment-chemprop-gpu.yml")
    local = _pip_dependencies(ROOT / "environment-chemprop.yml")

    assert CPU_TORCH_INDEX in local
    assert "torch==2.6.0+cpu" in local
    assert GPU_TORCH_INDEX not in local
    assert "torch==2.6.0+cu124" not in local
    assert gpu != local


def test_shared_chemprop_requirements_are_platform_neutral() -> None:
    requirements = [
        line.strip()
        for line in (ROOT / "requirements-chemprop.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert "chemprop==2.3.1" in requirements
    assert "lightning==2.6.5" in requirements
    assert not any(requirement.startswith("torch==") for requirement in requirements)
    assert not any("cu124" in requirement or "2.13.0" in requirement for requirement in requirements)
