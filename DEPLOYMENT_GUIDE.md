# 🚀 Deployment Guide

## Quick Start - Test Locally

The app is currently running at: **http://localhost:8501**

### What You Have:
✅ Professional Streamlit web app (`app.py`)
✅ All trained models in `FinalModel/`
✅ Dependencies installed
✅ README.md documentation
✅ .gitignore configured

---

## 📦 What's Completed

### ✅ Project Deliverables Status:

| Requirement | Status | Location |
|------------|--------|----------|
| **LaTeX Report** | ✅ Complete | `Rapport/rapport.tex` |
| **Scientific Poster** | ✅ Template Ready | `Rapport/poster.tex` |
| **Trained Models** | ✅ Complete | `FinalModel/models_4class/` |
| **Web Application** | ✅ Running | `app.py` (http://localhost:8501) |
| **README** | ✅ Complete | `README.md` |
| **Git Repository** | 🔄 Next Step | Need to create repo |
| **Cloud Deployment** | 🔄 Next Step | Deploy to Streamlit Cloud |
| **QR Code** | 🔄 Next Step | Generate after deployment |

---

## 🌐 Deploy to Streamlit Cloud (FREE)

### Step 1: Create GitHub Repository

1. **Go to GitHub**: https://github.com/new
2. **Create new repository**:
   - Name: `news-article-classifier`
   - Description: "Automated news classification using ML - ITBS Mini-Project"
   - Public repository
   - Don't initialize with README (we have one)

### Step 2: Push Code to GitHub

Open PowerShell in your project folder and run:

```powershell
# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: News classifier with Streamlit app"

# Link to your GitHub repo (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/news-article-classifier.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud

1. **Go to**: https://share.streamlit.io
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Fill in**:
   - Repository: `YOUR_USERNAME/news-article-classifier`
   - Branch: `main`
   - Main file path: `app.py`
5. **Click "Deploy"**
6. **Wait 2-3 minutes** for deployment
7. **Your app URL**: `https://YOUR_USERNAME-news-article-classifier.streamlit.app`

---

## 📱 Generate QR Code

After deployment, create QR code:

```python
# Install QR code library
pip install qrcode pillow

# Run this Python script
import qrcode

# Replace with your deployed URL
url = "https://YOUR_USERNAME-news-article-classifier.streamlit.app"

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(url)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")
img.save("qr_code.png")
print("✅ QR code saved as qr_code.png")
```

---

## 📊 Update Poster with Deployment Link

After deployment, update `Rapport/poster.tex`:

1. Replace `YOUR_DEPLOYED_URL_HERE` with your actual Streamlit URL
2. Add QR code image: `\includegraphics[width=0.15\textwidth]{qr_code.png}`
3. Recompile poster: `pdflatex poster.tex`

---

## ✅ Final Checklist

Before submission, ensure:

- [ ] GitHub repository is public
- [ ] README.md has correct links
- [ ] Streamlit app is live and working
- [ ] QR code generated and tested
- [ ] Poster includes QR code
- [ ] LaTeX report compiled (PDF)
- [ ] All figures generated and included
- [ ] Git repository has clean commit history

---

## 🎓 Submission Package

Your final submission should include:

1. **GitHub Repository URL**: `https://github.com/YOUR_USERNAME/news-article-classifier`
2. **Live Demo URL**: `https://YOUR_USERNAME-news-article-classifier.streamlit.app`
3. **LaTeX Report PDF**: `Rapport/rapport.pdf`
4. **Poster PDF**: `Rapport/poster.pdf`
5. **QR Code Image**: `qr_code.png`

---

## 🆘 Troubleshooting

### App won't start locally?
```bash
streamlit run app.py
```

### Module not found errors?
```bash
pip install -r requirements.txt
```

### Git push fails?
```bash
# Configure git if first time
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Streamlit Cloud deployment fails?
- Check `requirements.txt` is in root folder
- Ensure `FinalModel/` folder is committed
- Check app.py file paths are correct

---

## 📞 Support

If you encounter issues:
1. Check the error message in terminal/Streamlit Cloud logs
2. Verify all files are committed to Git
3. Test locally before deploying
4. Check Streamlit Cloud logs for deployment errors

---

## 🎉 Next Steps

1. **Test locally**: Open http://localhost:8501
2. **Create GitHub repo**: Follow Step 1 above
3. **Push to GitHub**: Follow Step 2 above
4. **Deploy to Streamlit**: Follow Step 3 above
5. **Generate QR code**: Use the Python script above
6. **Update poster**: Add QR code and URL
7. **Compile final PDFs**: LaTeX report + poster
8. **Submit**: Share repo link + live demo + PDFs

---

**Good luck with your presentation! 🚀**
