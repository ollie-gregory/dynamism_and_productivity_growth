# Has slower business dynamism been associated with slower productivity growth for U.S. sectors?

In short the answer is no. In long, the answer is not really, but I guess it has for some. What I mostly focus on here though is the potential reasons why slower business dynamism has not been associated with slower productivity growth.

## Repository structure

```
├── README.md
├── apa.csl # APA citation style file
├── data
│   ├── compustat.csv # Cleaned compustat data
│   ├── markup_and_productivity.csv # Cleaned markup and productivity data
│   ├── productivity_and_dynamism.csv # Cleaned productivity and dynamism data
│   ├── real_rnd_patents.csv # Cleaned real R&D and patents data
│   └── raw # Raw data files (duh)
│       ├── bds_4naics.csv
│       ├── bls_data.xlsx
│       ├── compustat_raw.csv
│       ├── naics_output_post97.csv
│       └── naics_output_pre97.csv
├── figs # This is for the figures in the paper
│   ├── fig1.png
│   ├── fig2.png
│   ├── fig3.png
│   └── table1.csv
├── index.html # HTML rendering for the website.
├── index.pdf # This is the PDF rendering.
├── index.qmd # This is the Quarto markdown file that renders the HTML and PDF.
├── notebooks # This is where most of the code is.
│   ├── analysis.ipynb # This was my initial exploratory analysis.
│   ├── cleaning.ipynb # This was the data cleaning notebook that produces the cleaned data from the raw data.
│   └── src
│       ├── figures.py # Running this produces the figures (run it from the repository root)
│       └── productivity_estimation.py # This is the productivity estimator class -- it took FOREVER to write.
└── references.bib # BibTeX file for references
```

## How to recreate this

Well, technically everything is already here. But if you wanted to run the code to get it to this point, you would need to do the following:

1. Make sure you have Python installed. I used Python 3.13, but as far as I can tell, these scripts should work with most modern versions of Python.
2. Install the required packages. You can do this by running `pip install -r requirements.txt` in the root directory of the repository. You will also need to install Quarto, which you can do by following the instructions at [quarto.org](https://quarto.org/docs/get-started/). If you have a mac with homebrew installed, you can run `brew install --cask quarto`.
3. Open the cleaning notebook `notebooks/cleaning.ipynb` and run it. This will produce the cleaned data files in the `data` directory.
4. Run the `notebooks/src/figures.py` script to produce the figures in the `figs` directory.
5. Render the `index.qmd` file by running `quarto render index.qmd` in the root directory of the repository. This will produce the HTML and PDF files in the root directory.
6. Open the `index.html` file in your web browser to view the website.

## Final note

This is a really interesting topic and I hope you enjoy exploring my code and the report! I may add more to this repository in the future, so feel free to check back.