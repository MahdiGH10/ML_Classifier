"""
QR Code Generator for Streamlit Deployment
Run this after deploying your app to generate QR code
"""

import qrcode
from PIL import Image

# ============================================
# CONFIGURATION - UPDATE THIS WITH YOUR URL
# ============================================
DEPLOYMENT_URL = "https://YOUR_USERNAME-news-article-classifier.streamlit.app"
OUTPUT_FILE = "qr_code.png"

# ============================================
# Generate QR Code
# ============================================

print("=" * 60)
print("📱 QR CODE GENERATOR")
print("=" * 60)

# Create QR code
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction
    box_size=10,
    border=4,
)

qr.add_data(DEPLOYMENT_URL)
qr.make(fit=True)

# Create image
img = qr.make_image(fill_color="black", back_color="white")

# Save
img.save(OUTPUT_FILE)

print(f"\n✅ QR Code generated successfully!")
print(f"📁 Saved as: {OUTPUT_FILE}")
print(f"🔗 URL: {DEPLOYMENT_URL}")
print("\n📋 Next steps:")
print("   1. Test the QR code with your phone")
print("   2. Add to your poster: \\includegraphics{qr_code.png}")
print("   3. Add to README.md")
print("=" * 60)
