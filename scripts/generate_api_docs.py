#!/usr/bin/env python3
"""Generate API documentation pages for MkDocs."""

from pathlib import Path


def generate_api_page(module_path: str, title: str) -> str:
    """Generate an API documentation page.
    
    Parameters
    ----------
    module_path : str
        The Python module path to document
    title : str
        The title for the documentation page
        
    Returns
    -------
    str
        The formatted markdown content for the API page
    """
    return f"""# {title}

::: {module_path}
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
      members_order: source
      show_signature_annotations: true
      show_if_no_docstring: true
      separate_signature: true
"""


def main():
    """Generate all API documentation pages."""
    # Define API pages to generate
    api_pages = {
        # Core modules
        "docs/api/core/base.md": ("leeq.core.base", "Core Base Classes"),
        "docs/api/core/elements.md": ("leeq.core.elements", "Quantum Elements"),
        "docs/api/core/engine.md": ("leeq.core.engine", "Execution Engine"),
        "docs/api/core/primitives.md": ("leeq.core.primitives", "Quantum Primitives"),
        
        # Experiments modules
        "docs/api/experiments/builtin.md": ("leeq.experiments.builtin", "Built-in Experiments"),
        "docs/api/experiments/base.md": ("leeq.experiments.base", "Experiment Base Classes"),
        
        # Theory modules
        "docs/api/theory/simulation.md": ("leeq.theory.simulation", "Simulation"),
        "docs/api/theory/tomography.md": ("leeq.theory.tomography", "Tomography"),
        "docs/api/theory/fits.md": ("leeq.theory.fits", "Fitting Functions"),
        "docs/api/theory/clifford.md": ("leeq.theory.clifford", "Clifford Gates"),
        
        # Compiler modules
        "docs/api/compiler/lbnl_qubic.md": ("leeq.compiler.lbnl_qubic", "LBNL QubiC Compiler"),
        "docs/api/compiler/base.md": ("leeq.compiler.base", "Compiler Base Classes"),
    }
    
    # Get base path from script location
    base_path = Path(__file__).parent.parent
    
    # Generate each API page
    for page_path, (module, title) in api_pages.items():
        full_path = base_path / page_path
        
        # Ensure parent directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate and write the content
        content = generate_api_page(module, title)
        full_path.write_text(content)
        print(f"Generated {page_path}")
    
    print(f"\nSuccessfully generated {len(api_pages)} API documentation pages")


if __name__ == "__main__":
    main()