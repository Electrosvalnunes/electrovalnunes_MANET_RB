# Publishing this repository on GitHub

## 1. Copy the missing datasets

```bash
cd "/media/osvaldo/UBUNTU 22_0/CAMPIN_SEND_1807/fanet-bayesian-ids-portability"

cp "/media/osvaldo/UBUNTU 22_0/CAMPIN_SEND_1807/Dataset_electrosvalnunes_manet_v2.csv" data/
cp "/media/osvaldo/UBUNTU 22_0/CAMPIN_SEND_1807/ns3_like_packet_loss_causes_v1_50k.csv" data/
```

## 2. Inspect the repository

```bash
find . -maxdepth 4 -type f | sort
du -h data/*.csv
```

## 3. Initialize Git

```bash
git init -b main
git add .
git status
git commit -m "Initial release: FANET Bayesian IDS portability framework"
```

## 4. Create an empty GitHub repository

Suggested repository name:

```text
fanet-bayesian-ids-portability
```

Do not initialize the remote repository with another README, `.gitignore`, or
license when creating it.

## 5. Connect and push

Replace `USERNAME` with the GitHub username:

```bash
git remote add origin https://github.com/USERNAME/fanet-bayesian-ids-portability.git
git push -u origin main
```

## 6. Large dataset option

If either CSV is too large for normal Git tracking, do not commit it first.
Use a research-data repository such as Zenodo and replace the CSV with a DOI and
download instructions, or configure Git LFS before the first dataset commit.

Example Git LFS commands:

```bash
git lfs install
git lfs track "data/*.csv"
git add .gitattributes data/*.csv
git commit -m "Add datasets with Git LFS"
git push
```

## 7. Final checks

- replace `USERNAME` in `CITATION.cff`;
- choose the repository visibility;
- choose appropriate licenses;
- confirm the external dataset redistribution terms;
- create a tagged release such as `v1.0.0`.
