# utils/eval_ragas.py

import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness, AnswerRelevancy,
    AnswerCorrectness, ContextPrecision, ContextRecall
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import logging
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()
print("✔ Env loaded from .env")
import os

def fmt(v, d=3):
    try:
        return round(float(v), d)
    except:
        return 0.0


def run_ragas_evaluation():
    """
    Menjalankan evaluasi RAGAS:
      - context precision
      - context recall
      - answer relevancy
      - faithfulness
      - answer correctness

    Return:
      ragas_item (list per item)
      ragas_agg (aggregate)
    """

    logger.info("\n==============================")
    logger.info("📊  RAGAS Evaluation Started -v1")
    logger.info("==============================\n")

    openai_key = os.getenv("OPENAI_API_KEY")
    # ---------------------------------
    # SETUP LLM DAN EMBEDDER
    # ---------------------------------
    ragas_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=openai_key
    )
    ragas_embed = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_key
    )
    ragas_embed.client.batch_size = 128


    # ---------------------------------
    # DATASET RAGAS
    # ---------------------------------
    data = {
        "question": [
            "Siapa yang disebut sebagai desainer pengalaman pengguna?"
        ],
        "answer": [
            "Seorang desainer pengalaman pengguna adalah seseorang yang menyelidiki dan menganalisis bagaimana perasaan pengguna terhadap produk yang ia tawarkan."
        ],
        "ground_truth": [
            "Seorang desainer pengalaman pengguna adalah seseorang yang menyelidiki dan menganalisis bagaimana perasaan pengguna terhadap produk yang ia tawarkan."
        ],
        "contexts": [
            [
                """
- Kompleksitas dan Persepsi

Desain pengalaman pengguna berfokus pada bagaimana merancang pengalaman terbaik ketika seseorang menggunakan sebuah layanan atau produk. Bidang ini dapat mencakup berbagai jenis layanan dan produk, termasuk desain pameran di museum. Meskipun cakupannya luas, istilah desain pengalaman pengguna paling sering digunakan dalam konteks situs web, aplikasi berbasis web, dan aplikasi perangkat lunak. Sejak paruh kedua dekade pertama abad ini, teknologi berkembang semakin kompleks, dan fungsi aplikasi maupun situs web menjadi jauh lebih luas dan rumit. Pada awal kemunculannya, situs web hanyalah halaman statis sederhana yang menyajikan informasi bagi pengguna yang ingin mencari sesuatu. Namun, beberapa dekade kemudian, kita menemukan berbagai situs yang interaktif dan mampu menghadirkan pengalaman yang jauh lebih kaya bagi para penggunanya. Anda dapat menambahkan berbagai fitur dan fungsi apa pun ke sebuah situs atau aplikasi, tetapi keberhasilan sebuah proyek tetap bergantung pada satu faktor utama: bagaimana perasaan pengguna terhadapnya. “Manusia selalu bersifat emosional dan selalu bereaksi secara emosional terhadap artefak dalam dunia mereka.”
—Alan Cooper, President of Cooper
Pertanyaan yang menjadi perhatian para desainer pengalaman pengguna meliputi hal hal berikut:
• Apakah situs atau aplikasi tersebut memberikan nilai bagi pengguna? • Apakah pengguna merasa situs atau aplikasi tersebut mudah digunakan dan mudah dinavigasi? • Apakah pengguna benar benar menikmati pengalaman menggunakan situs atau aplikasi tersebut? Seorang desainer pengalaman pengguna dapat mengatakan bahwa ia telah melakukan pekerjaannya dengan baik ketika semua pertanyaan tersebut dapat dijawab dengan “Ya.”
Apa itu Pengalaman Pengguna (UX)? Secara umum, pengalaman pengguna menggambarkan bagaimana perasaan seseorang ketika menggunakan sebuah produk atau layanan. Dalam banyak kasus, produk tersebut berupa situs web atau aplikasi. Setiap bentuk interaksi antara manusia dan objek selalu memiliki pengalaman pengguna terkait, namun pada umumnya para praktisi UX berfokus pada hubungan antara pengguna manusia dan komputer serta produk berbasis komputer seperti situs web, aplikasi, dan sistem. Apa itu Desainer UX? Seorang desainer pengalaman pengguna adalah seseorang yang menyelidiki dan menganalisis bagaimana perasaan pengguna terhadap produk yang ia tawarkan. Desainer UX kemudian menerapkan wawasan tersebut dalam proses pengembangan produk untuk memastikan pengguna memperoleh pengalaman terbaik saat menggunakan produk tersebut. Desainer UX melakukan berbagai kegiatan, seperti melakukan riset, menganalisis temuan, menyampaikan hasil temuan kepada anggota tim pengembangan, memantau jalannya proyek untuk memastikan temuan tersebut diterapkan, dan banyak tugas lainnya. Mengapa UX Penting? Dulu, desain produk cenderung sederhana. Para desainer membuat sesuatu yang menurut mereka keren dan berharap klien menyukainya. Namun, ada dua masalah utama dengan pendekatan tersebut. Pertama, persaingan untuk mendapatkan perhatian pengguna jauh lebih sedikit pada masa itu. Kedua, tidak ada pertimbangan terhadap pengguna produk sama sekali. Keberhasilan atau kegagalan sebuah proyek desain lebih banyak ditentukan oleh keberuntungan dan penilaian subjektif tim desain. Fokus pada pengalaman pengguna membuat proses desain lebih berorientasi pada kebutuhan manusia. ...
- Tokoh besar dalam bidang pengalaman pengguna, Don Norman, yang juga mencetuskan istilah User Experience, menjelaskan apa itu design thinking dan apa keistimewaannya:
“…semakin saya merenungkan hakikat desain dan melihat pengalaman saya bersama para insinyur, pelaku bisnis, dan orang lainnya yang memecahkan masalah tanpa mempertanyakan atau mengkajinya lebih jauh, saya menyadari bahwa mereka dapat memperoleh manfaat dari pemikiran desain. Desainer telah mengembangkan sejumlah teknik untuk menghindari terjebak pada solusi yang terlalu mudah. Mereka memperlakukan masalah awal sebagai saran, bukan pernyataan final, lalu berpikir secara luas tentang isu sebenarnya yang mungkin tersembunyi di balik pernyataan masalah tersebut, misalnya dengan menggunakan pendekatan ‘Five Whys’ untuk menemukan akar masalah.”
—Don Norman, Rethinking Design Thinking
Design Thinking sebagai Alat Penting dan Jalan Ketiga
Proses desain sering melibatkan banyak kelompok dari berbagai departemen. Karena itu, mengembangkan, mengategorikan, dan mengatur ide serta solusi masalah bisa menjadi tantangan. Salah satu cara untuk menjaga agar proyek desain tetap berjalan dengan baik dan terstruktur adalah menggunakan pendekatan design thinking. Tim Brown, CEO perusahaan inovasi dan desain IDEO, menjelaskan dalam bukunya Change by Design bahwa design thinking berlandaskan pemahaman holistik dan empatik terhadap masalah yang dihadapi manusia. Pendekatan ini menyertakan konsep yang bersifat ambigu atau subjektif seperti emosi, kebutuhan, motivasi, dan pendorong perilaku. Pendekatan tersebut berbeda dengan metode ilmiah murni yang cenderung menjaga jarak dalam memahami serta menguji kebutuhan dan emosi pengguna melalui riset kuantitatif. Tim Brown merangkum bahwa design thinking merupakan jalan ketiga. Design thinking pada dasarnya adalah pendekatan pemecahan masalah yang terbentuk dalam dunia desain, yang menggabungkan perspektif holistik berpusat pada pengguna dengan riset rasional dan analitis untuk menghasilkan solusi inovatif. Sains dan Rasionalitas dalam Design Thinking
Beberapa aktivitas ilmiah dalam design thinking mencakup analisis mengenai bagaimana pengguna berinteraksi dengan produk dan meneliti kondisi penggunaan produk tersebut. Aktivitas ini mencakup penelitian kebutuhan pengguna, menggabungkan pengalaman dari proyek sebelumnya, mempertimbangkan kondisi saat ini dan masa depan yang relevan dengan produk, menguji parameter masalah, serta menguji penerapan solusi alternatif. Tidak seperti pendekatan ilmiah murni yang biasanya menguji sebagian besar karakteristik atau variabel yang sudah diketahui untuk menemukan solusi, penyelidikan dalam design thinking juga memasukkan unsur unsur ambigu. Tujuannya adalah menemukan parameter baru yang sebelumnya tidak diketahui dan membuka strategi alternatif. Setelah menghasilkan berbagai kemungkinan solusi, proses pemilihannya ditopang oleh rasionalitas. Para desainer didorong untuk menganalisis dan menguji kebenaran solusi solusi tersebut sehingga dapat menemukan pilihan terbaik untuk setiap masalah atau hambatan yang teridentifikasi di setiap fase proses desain. Dengan pemahaman ini, lebih tepat jika dikatakan bahwa design thinking bukan sebatas berpikir di luar kotak, melainkan berpikir pada tepinya, sudutnya, lipatannya, bahkan di bawah kode batangnya, seperti yang dikatakan oleh Clint Runge. (Clint Runge adalah Founder dan Managing Director Archrival, sebuah agensi pemasaran anak muda, sekaligus profesor di University of Nebraska-Lincoln.)
                """
            ]
        ]
    }

    ds = Dataset.from_dict(data)
    n = len(ds)
    # ds = Dataset.from_dict({
    #     "question": questions,
    #     "answer": predictions,
    #     "contexts": contexts_used,
    #     "ground_truth": clean_references
    # })

    metrics = [
        ContextPrecision(),
        ContextRecall(),
        AnswerRelevancy(),
        Faithfulness(),
        AnswerCorrectness()
    ]
    import logging
    logging.getLogger("ragas").setLevel(logging.DEBUG)
    logging.getLogger("ragas.metrics").setLevel(logging.DEBUG)
    logging.getLogger("ragas.llms").setLevel(logging.DEBUG)
    logging.getLogger("ragas.prompt").setLevel(logging.DEBUG)



    scores = evaluate(ds, metrics, ragas_llm, ragas_embed)

   
    ragas_item = []


    logger.info("🔍 Detail skor per item:\n")

    for i in range(1):

        q = ds[i]["question"]
        a = ds[i]["answer"]
        ctx = ds[i]["contexts"]
        gt = ds[i]["ground_truth"]



        item_scores = {
            "context_precision": fmt(scores["context_precision"][i]),
            "context_recall": fmt(scores["context_recall"][i]),
            "answer_relevancy": fmt(scores["answer_relevancy"][i]),
            "faithfulness": fmt(scores["faithfulness"][i]),
            "answer_correctness": fmt(scores["answer_correctness"][i]),
        }
        ragas_item.append(item_scores)

        logger.info(f"\n==============================")
    
        logger.info("==============================")
        logger.info(f"Question       : {q}")
        logger.info(f"Answer         : {a}")
        logger.info(f"Ground Truth   : {gt}")
        logger.info(f"Contexts       : {ctx}")
        logger.info("--- Skor ---")
        logger.info(f"  context_precision : {item_scores['context_precision']}")
        logger.info(f"  context_recall    : {item_scores['context_recall']}")
        logger.info(f"  answer_relevancy  : {item_scores['answer_relevancy']}")
        logger.info(f"  faithfulness      : {item_scores['faithfulness']}")
        logger.info(f"  answer_correctness: {item_scores['answer_correctness']}")
        logger.info("")

    # ---------------------------------
    # AGGREGATE
    # ---------------------------------
    ragas_agg = {
        "context_precision": fmt(np.mean(scores["context_precision"])),
        "context_recall": fmt(np.mean(scores["context_recall"])),
        "answer_relevancy": fmt(np.mean(scores["answer_relevancy"])),
        "faithfulness": fmt(np.mean(scores["faithfulness"])),
        "answer_correctness": fmt(np.mean(scores["answer_correctness"])),
    }

    logger.info("\n==============================")
    logger.info("📈  RAGAS Aggregate Scores")
    logger.info("==============================\n")

    for k, v in ragas_agg.items():
        print(f"  {k:20s}: {v}")

    logger.info("\n✔ Evaluasi RAGAS selesai.\n")

    return ragas_item, ragas_agg

if __name__ == "__main__":
    run_ragas_evaluation()