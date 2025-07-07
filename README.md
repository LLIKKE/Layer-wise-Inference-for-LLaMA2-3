# Layer-wise Inference for LLaMA2/3 with Wikitext2 PPL Evaluation

This project enables **layer-by-layer loading and inference of LLaMA2/LLaMA3 models on GPU**, followed by **evaluation on the Wikitext2 dataset using Perplexity (PPL)**.

---

## 🔍 Why This Project?

Running large language models (LLMs) like LLaMA2/3 often requires massive GPU memory. This project:

- Supports **inference of models as large as 70B on a single 24GB GPU** by loading layers one at a time.
- Enables **fast prototyping of optimization techniques** like **KV cache pruning** and other LLM enhancements.
- Provides a lightweight framework for evaluating model performance using PPL without full model loading.

---

## 🚀 How to Use

1. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
   
   ```bash
   transformers==4.52.4
   ```

3. Run the evaluation:

   ```bash
   python main.py --input_model /data/meta-llama/Llama-2-7b-hf
   ```

> Replace the `--input_model` path with your actual model directory.

---

## 📊 Evaluation Results (Wikitext2 PPL)

| Model            | PPL   |
|------------------|-------|
| LLaMA2 7B        | 5.468 |
| LLaMA2 7B Chat   | 6.94  |
| LLaMA2 13B       | 4.88  |
| LLaMA2 70B       | 3.32  |
| LLaMA3 8B        | 6.16  |
| LLaMA3 70B       | 2.86  |

---

## 🧩 Use Cases

- Run LLaMA models on GPUs with **< 24GB memory**
- **Develop and test inference-time optimizations**
- Benchmark **PPL efficiently on Wikitext2**
- Ideal for research, experimentation, and educational purposes

