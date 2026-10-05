"""Text normalization shared by parser lookup and semantic execution."""
import re

def normalize_text(s):
    return re.sub(r'\s+', ' ', str(s).strip()).strip(' \t\r\n\"\'').lower()
