 # KEPLER EXOPLANET CLASSIFICATION

> Using machine learning to ditinguish confirmed exoplanets from false positives
> In NASA's Kepler Space Telescope Dataset

---

## Project Overview
NASA's Kepler mission monitored over 150,000 stars for tiny dips in brightness
caused by orbiting planets passing in front of their host star. Each candidate
planet - ca;;ed a Kepler Object Of Interest (KOI) - was then classified as
CONFIRMED, FLASE POSITIVE, or CANDIDATE based on the follow-up analysis

This project applies Exploratory Data Analysis, statistical testing, and machine
learning to the cumulative Kepler dataset (9,564 objects, 50 features) to:

 - Understand the physical properties that distinguish real planets from flase detection
 - Build a classification model that predicts whether a KOI is a confirmed planet
 - Deploy an interactive dashboard for exploring the dataset visually

---

## Key Findings
 - **Class imbalance**: 52.5% of the objects are FALSE POSITIVE, 24% CONFIRMED, 23.5 CANDIDATE
 - **Planet Radius**: ('koi_prad') is the strongest single predictor - confirmed planet cluster
    below 4 Earth Radii, while false positives are spread widely
 - **Insolation flux** and **disposition score** are the second and third most predictive features
 - **XGBoost acheived ROC-AUC of 0.XX** on the held-out test set, outperforming the Random Forest baseline of 0.XX
 - Confirmed Planets and flase positives are statistically significantly different across all 12 key features (t-test p < 0.001 for each)

---

## Project Structure
```
kepler-exoplanet-classification/
│
├── data/
│   ├── cumulative.csv          # Raw Kepler dataset (NASA)
│   └── kepler_clean.csv        # Cleaned dataset (output of Phase 1)
│
├── phase1_eda.py               # Data loading, cleaning, distributions
├── phase2_statistics.py        # Hypothesis tests, correlations, outliers
├── phase3_visualizations.py    # Static + interactive charts, Streamlit app
├── phase4_modeling.py          # ML models, evaluation, feature importance
├── phase5_portfolio.py         # README generator, notebook guide
│
├── kepler_dashboard.py         # Streamlit dashboard (standalone)
│
│
├── outputs/                    # All saved charts and HTML files
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech stack

| Category       | Libraries                              |
|----------------|----------------------------------------|
| Data handling  | pandas, numpy                          |
| Visualization  | matplotlib, seaborn, plotly            |
| Statistics     | scipy                                  |
| ML models      | scikit-learn, xgboost                  |
| Imbalance      | imbalanced-learn (SMOTE)               |
| Dashboard      | streamlit                              |

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone hhtps://github.com/aditya-patra1011/kepler-exoplanet-classification
cd kepler-exoplanet-classification
```

### 2. Install Dependcies
```bash
pip install -r requirements.txt
```

### 3. Run the analysis phases in order
```bash
python phase1_eda.py
python phase2_statistics.py
python phase3_visualizations.py
python phase4_modeling.py
```
### 4. Launch the interactive dashboard
```bash
streamlit run kepler_dashboard.py
```

### 5. Or open the full jupyter notebook
```bash
jupyter notebooks/kepler_full_analysis.ipynb
```
---

## Dataset
- **Source**: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- **Also on**: [Kaggle- Kepler Exoplanet Search Results](https://www.kaggle.com/datadets/nasa/kepler-exoplanet-search-results)
- **Rows**: 9,564 Kepler Objects of Interest
- **Features**: 50 columns including orbital, planetary, and stellar measurements

---

## Author
**M. Aditya Patra**
- LinkedIn: [linkedin.com/in/M. Aditya Patra](www.linkedin.com/in/m-aditya-patra-1339853b9)
- GitHub: [github.com/aditya-patra1011](https://github.com/aditya-patra1011)
- Portfolio: [portfolio-three-theta-vwkfspsq2q.vercel.app/](https://portfolio-three-theta-vwkfspsq2q.vercel.app/)

---

## License
MIT License - feel free to use and adapt this project
