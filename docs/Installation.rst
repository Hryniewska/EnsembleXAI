Installation Guide
==================

User Installation
-----------------
Usage of a virtual environment is strongly encouraged.

#. Clone the repository from github.
#. Install the requirements in commandline.

    #. Activate virtual environment if applicable.
    #. `pip install -r /correct_path/EnsembleXAI/requirements.txt`.

Downloading as a package (currently unavailable due to repository currently being private).

`pip install git+https://github.com/anonymous-conference-journal/EnsembleXAI#egg=EnsembleXAI&subdirectory=EnsembleXAI`

Package Development Additional Installation
-------------------------------------------
To run documentation creation additional packages are required:

#. Nbsphinx extension to convert notebooks to html requires seperate installation of `pandocs`, installation with conda is recommended.
#. Optionally activate the virtual environment.
#. Run `pip install sphinx sphinx_rtd_theme nbsphinx`.
#. Optionally for a clean install run `.\\docs\\make clean`.
#. Then in the project directory run: `.\\docs\\make html`.