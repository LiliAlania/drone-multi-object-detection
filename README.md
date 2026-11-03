# UCU AI Homework Report Template

Your report is published as a website in UCU colors. The page is generated from `report.md`, and it shows the date of your repository's last push.

## How to fill

1. Click **Use this template** to create your own **public** repository. If you fork instead, also open the **Actions** tab and enable workflows.
2. Enable GitHub Pages (see [below](#enabling-github-pages)).
3. Edit **`report.md`**:
   - Fill in the header block at the top (`title`, `author`, `course`, `date`).
   - Write the report in Markdown. You can use tables, code blocks, and LaTeX math (`$...$` inline, `$$...$$` for display).
   - Put images in `images/` and reference them as `![caption](images/plot.png)`.
4. Commit and push to `main`. GitHub Actions deploys the site automatically within about a minute, at `https://<username>.github.io/<repo>/`.
5. Submit that link.

## Enabling GitHub Pages

This is a one-time step. GitHub does not copy it to your repository.

1. In your repository, open **Settings → Pages**.
2. Under **Build and deployment → Source**, choose **GitHub Actions**. It saves as soon as you choose it.
3. Open the **Actions** tab, select the failed **Deploy report** run, and click **Re-run jobs**.

Until you do this, every deploy fails with the error "GitHub Pages is not set up".

## Local preview

```bash
python -m http.server 8000
# open http://localhost:8000
```

Opening `index.html` directly from disk will not work.

## VisDrone EDA and dataset subsets

The website/report instructions above apply to the course report. The EDA work lives in
separate `src/`, `notebooks/`, `tests/`, `results/`, `figures/`, and `splits/` folders;
it does not modify the report website.

### Run the EDA in Google Colab

1. Open [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) in Colab (upload it or open it
   from the public GitHub repository).
2. Set `MOUNT_DRIVE = True` in the setup cell if you want to mount Drive. The notebook
   clones this repository into `/content/drone-multi-object-detection` when needed.
3. Keep the dataset at `/content/data` to skip downloading. Otherwise the notebook calls
   `src/get_data.py`, which uses the Kaggle CLI. Install/configure Kaggle credentials in
   Colab and accept the competition rules before running the download cell; no credentials
   are stored in this repository.
4. Run all cells. Results and figures are saved under `results/`, `figures/eda/`, and
   `splits/`. The notebook intentionally leaves its observations for the student to fill in.

The helpers are in `src/data_utils.py`. Run their synthetic-data tests from the repository
root with `python -m unittest discover -s tests`.

### Create a reusable subsampled dataset

From a Python environment with pandas and NumPy installed, run:

```bash
python -m src.make_subset --data-root /content/data --output-dir /content/visdrone-subset --step 5 --splits train val
```

The command writes a YOLO-layout directory and a `.zip` beside it. To copy the zip to
mounted Drive, add, for example,
`--drive-copy /content/drive/MyDrive/visdrone-subset.zip`. The step is applied independently
to each clip in frame order; ignored-region labels remain in a separate `ignore/` tree.

## ⚠️ Rules

You may change **only** `report.md` and the `images/` folder.

**Any change to the core website (`index.html`, `assets/`, `.github/`) will annul the grade for the homework.** This includes edits that change how the last-push date is shown.

The last-push date is read from GitHub and is used to check deadlines.
