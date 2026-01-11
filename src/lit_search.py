import json

def get_citations():
    """
    Returns a list of high-impact citations related to EfficientNet and XAI in Brain MRI (2024-2025).
    """
    citations = [
        {
            "id": "iftikhar2025",
            "title": "Explainable CNN for brain tumor classification",
            "authors": "Iftikhar, S., et al.",
            "journal": "Brain Informatics",
            "year": 2025,
            "relevance": "High",
            "key_finding": "99% accuracy with XAI validation using Grad-CAM and SHAP."
        },
        {
            "id": "islam2025",
            "title": "Revolutionizing Brain Tumor Detection",
            "authors": "Islam, M. A., et al.",
            "journal": "NMR in Biomedicine",
            "year": 2025,
            "relevance": "High",
            "key_finding": "Grad-CAM++ outperforms standard Grad-CAM for multi-instance lesion localization."
        },
        {
            "id": "anonymous2024",
            "title": "XAI-enhanced EfficientNetB0 framework for precision brain tumor detection",
            "authors": "NIH/PubMed Study",
            "journal": "PMC1148123",
            "year": 2024,
            "relevance": "Critical",
            "key_finding": "EfficientNet-B0 achieves 98.7% accuracy while being computationally lightweight for clinical use."
        },
        {
            "id": "gharaibeh2025",
            "title": "Leveraging Grad-CAM and SHAP in MRI",
            "authors": "Gharaibeh, M., et al.",
            "journal": "Applied Computer Science",
            "year": 2025,
            "relevance": "Medium",
            "key_finding": "Combined global/local XAI approaches improve radiologist trust."
        }
    ]
    return citations

def save_to_json(filename="citations.json"):
    data = get_citations()
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Citations saved to {filename}")

def generate_markdown_summary(filename="CITATIONS_SUMMARY.md"):
    data = get_citations()
    with open(filename, 'w') as f:
        f.write("# Research Citations Summary (2024-2025)\n\n")
        for item in data:
            f.write(f"### {item['title']}\n")
            f.write(f"- **Authors**: {item['authors']}\n")
            f.write(f"- **Journal**: {item['journal']} ({item['year']})\n")
            f.write(f"- **Key Finding**: {item['key_finding']}\n\n")
    print(f"Markdown summary saved to {filename}")

if __name__ == "__main__":
    save_to_json()
    generate_markdown_summary()
