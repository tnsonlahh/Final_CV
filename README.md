# CLIP-based Fashion Retrieval with Qdrant

This project implements a **fashion product retrieval system** using **CLIP embeddings** and **Qdrant vector database**.  
The system supports:

- Text → Image retrieval  
- Text → Image + Title (hybrid) retrieval  
- Local evaluation using metadata-based pseudo ground truth  
- Interactive UI with Gradio  

---
How to Run
1️⃣ Clone the repository
git clone https://github.com/tnsonlahh/Final_CV.git
cd Final_CV

2️⃣ Start Qdrant
docker compose up -d
Qdrant API: http://localhost:6333
Qdrant Dashboard: http://localhost:6333/dashboard

3️⃣ Install Python dependencies
pip install -r requirements.txt

4️⃣ Preprocess metadata
Open and run:
preprocessing_metadata.ipynb

5️⃣ Upload data to Qdrant
python upload_to_qdrant.py

6️⃣ Run local retrieval pipeline
python main.py


Input: text query
Output: top-K retrieved images from Qdrant

7️⃣ Run Gradio demo UI
python gradio_app.py


Open the Gradio link printed in terminal to try interactive search.
