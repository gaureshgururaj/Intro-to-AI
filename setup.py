from setuptools import setup, find_packages

setup(
    name = 'svlearn-ai-intro',
    version = '0.1',
    packages = find_packages(where='src'),
    package_dir = {'': 'src'},

    python_requires = '>=3.10',

    install_requires = [
        # DOCUMENTATION BUILDERS
            
        # mkdocstrings[python] -- Generates documentation from docstrings
        'mkdocstrings[python]',
        # mkdocs-material -- Packages mkdocs with several plugins
        'mkdocs-material',
        # mkdocs-material-extensions -- Adds additional features to mkdocs-material
        'mkdocs-material-extensions',
        # mkdocs-awesome-pages-plugins -- Allows Navigation to be generated from the filesystem
        'mkdocs-awesome-pages-plugin',
        # mknotebooks -- Allows Jupyter Notebooks to be included in the documentation
        'mknotebooks',
        # auto-link for mkdocs simplifies the hyperlinks
        'mkdocs-autolinks-plugin',
        # table-reader plugin
        'mkdocs-table-reader-plugin',
        # mkdocs-mermaid2-plugin -- Adds support for Mermaid diagrams
        'mkdocs-mermaid2-plugin',
        # mkdocs-include-markdown-plugin -- allows include content of different codes
        'mkdocs-include-markdown-plugin', 

        # for notebook widgets
        'ipywidgets' ,

        # Decorator | For creating decorators
        'decorator',

        # Pathlib | For handling file paths
        'pathlib',

        # PyKwalify | For validating YAML files
        'pykwalify',

        # Pytest | For testing
        'pytest',

        # Rich | Rich Console Output
        'rich',

        # Torch | PyTorch
        'torch',

        # Skorch
        'skorch',

        # librosa | for audio preprocessing
        'librosa',

        # A great library for logging
        'loguru',

        # pytorch stuff (on command line while doing pip install, also include --index-url https://download.pytorch.org/whl/cu124)
        'torch', 'torchvision', 'torchaudio',

        # matplotlib
        'matplotlib',

        # common libraries required in the supportvectors-common.ipynb
        'numpy', 'pandas', 'scikit-learn', 'scipy', 'seaborn', 'altair', 'plotly', 
        
        # autoencoders
        'pythae',

        # transformers
        'transformers' , 'evaluate' , 'datasets',
        
        # sentence transformers
        'sentence-transformers',

        # Diffusion models
        'diffusers','accelerate','peft',
        
        # KAN
        'pykan',
        
])
