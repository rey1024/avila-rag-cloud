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
    model="gpt-4o-mini"
    print(f"Using LLM model: {model}")
    ragas_llm = ChatOpenAI(
        model=model,
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
            "Kembangkan sebuah panduan untuk pemula yang ingin mempelajari dasar-dasar desain pengalaman pengguna, dengan memasukkan elemen-elemen penting dari kursus 'User Experience: The Beginner’s Guide'."
        ],
        "answer": [
            """
Untuk pemula yang ingin mempelajari dasar-dasar desain pengalaman pengguna, berikut adalah panduan yang mencakup elemen-elemen penting dari kursus 'User Experience: The Beginner’s Guide':

1. **Keterampilan yang Diperlukan**: Pelajari keterampilan yang dibutuhkan untuk membantu perusahaan menghadirkan pengalaman pengguna yang tepat melalui pendekatan yang telah teruji.

2. **Output UX**: Pelajari cara menghasilkan berbagai output UX dengan bantuan templat yang dapat diunduh, mulai dari peta perjalanan pengguna hingga prototipe berbasis kertas.

3. **Membangun Portofolio**: Gunakan templat tersebut untuk membangun portofolio UX yang kuat, yang akan membantu saat melamar pekerjaan.

4. **Wawasan dari Ahli**: Dapatkan akses ke video wawancara bersama perekrut dan ahli UX dari perusahaan seperti Google dan SAP, yang memberikan wawasan mengenai keterampilan dan kualitas yang dicari.

Dengan mengikuti panduan ini, Anda dapat memulai perjalanan dalam desain pengalaman pengguna dengan pemahaman yang solid.
            """
        ],
        "ground_truth": [
            "Panduan untuk pemula yang ingin mempelajari dasar-dasar desain pengalaman pengguna dapat dimulai dengan mempelajari kursus 'User Experience: The Beginner’s Guide'. Kursus ini menawarkan panduan komprehensif untuk memulai karier di bidang desain UX. Anda akan mempelajari keterampilan yang dibutuhkan untuk membantu perusahaan menghadirkan pengalaman pengguna yang tepat melalui pendekatan yang telah teruji. Kursus ini juga menyediakan templat yang dapat diunduh untuk menghasilkan berbagai output UX, seperti peta perjalanan pengguna dan prototipe berbasis kertas. Selain itu, Anda akan mendapatkan wawasan dari video wawancara bersama perekrut dan ahli UX dari perusahaan seperti Google dan SAP, yang akan memberi Anda keunggulan dalam proses rekrutmen di bidang UX."
        ],
        "contexts": [
            [
                """
[77] Menghasilkan Ide dan Solusi Kreatif melalui Pemahaman Manusia secara Holistik
Dengan landasan yang kuat pada sains dan rasionalitas, design thinking berupaya membangun pemahaman holistik dan empatik terhadap masalah yang dihadapi manusia. Pendekatan ini mencoba memahami manusia secara mendalam, termasuk berbagai konsep yang bersifat ambigu atau subjektif seperti emosi, kebutuhan, motivasi, dan pendorong perilaku. Karakter proses menghasilkan ide dan solusi dalam design thinking membuat pendekatan ini lebih peka dan lebih tertarik pada konteks tempat pengguna beroperasi serta masalah atau hambatan yang mereka hadapi ketika berinteraksi dengan sebuah produk. Unsur kreatif dalam design thinking muncul dari metode yang digunakan untuk menghasilkan solusi dan memperoleh wawasan mengenai praktik, tindakan, serta pola pikir pengguna nyata. Design Thinking adalah Proses Iteratif dan Tidak Linear
Design thinking merupakan proses yang iteratif dan tidak linear. Artinya, tim desain terus menerus menggunakan hasil yang mereka peroleh untuk meninjau, mempertanyakan, dan meningkatkan asumsi awal, pemahaman, serta hasil sebelumnya. Temuan dari tahap akhir proses awal akan memperkaya pemahaman kita tentang masalah, membantu menentukan parameter masalah, memungkinkan kita mendefinisikan ulang masalah, dan yang paling penting memberikan wawasan baru sehingga kita dapat melihat berbagai solusi alternatif yang sebelumnya tidak terlihat pada tingkat pemahaman awal. Design Thinking: Panduan Pemula

Perusahaan perusahaan terkemuka dunia seperti Apple, Google, dan Samsung telah menggunakan pendekatan design thinking karena mereka memahami bahwa pendekatan ini menjadi jalan utama menuju inovasi dan keberhasilan produk. Melalui Design Thinking: The Beginner’s Guide, Anda akan mempelajari secara mendalam lima fase pendekatan pemecahan masalah yang mengubah paradigma ini, yaitu empathize, define, ideate, prototype, dan test. Dengan panduan rinci mengenai berbagai aktivitas pemecahan masalah, mulai dari teknik menghasilkan ide seperti brainstorming dan penggunaan analogi, hingga cara mengumpulkan umpan balik dari prototipe, Anda dapat mengunduh berbagai templat yang tersedia dan menggunakannya secara efektif dalam pekerjaan Anda. Bersiaplah untuk mempelajari, mengeksplorasi, dan menguasai design thinking sehingga dapat menjadi nilai tambah dan membuka tahap berikutnya dalam perjalanan profesional Anda. Tujuh Faktor yang Mempengaruhi Pengalaman Pengguna

Pengalaman pengguna memiliki peran penting dalam menentukan keberhasilan atau kegagalan sebuah produk di pasar. Namun, apa sebenarnya yang dimaksud dengan pengalaman pengguna? Banyak orang sering menyamakan pengalaman pengguna dengan usability, yaitu seberapa mudah suatu produk digunakan. Meskipun benar bahwa bidang pengalaman pengguna berawal dari konsep usability, cakupannya kini jauh lebih luas. Memperhatikan seluruh aspek pengalaman pengguna menjadi hal penting agar sebuah produk dapat berhasil di pasar. “Untuk menjadi desainer yang hebat, Anda perlu memahami lebih dalam bagaimana orang berpikir dan bertindak.”
— Paul Boag, Co-Founder Headscape Limited
Peter Morville, salah satu pelopor dalam bidang pengalaman pengguna yang menulis berbagai buku terlaris dan menjadi penasihat bagi banyak perusahaan Fortune 500, menyusun tujuh faktor yang menggambarkan pengalaman pengguna.
[72] Pendekatan ini meningkatkan peluang keberhasilan sebuah produk ketika memasuki pasar, terutama karena tidak mengandalkan asumsi bahwa pengguna otomatis menerima produk hanya karena nama merek tertentu. Di Mana Saja Desain UX Ditemukan? Desain pengalaman pengguna hadir dalam berbagai lingkungan proyek saat ini, antara lain:
• Proyek kompleks
Semakin rumit sebuah proyek, semakin penting nilai desain UX. Banyak fitur yang ditangani dengan cara yang salah dapat membuat pengguna merasa frustrasi. • Startup
Tim UX khusus mungkin tidak selalu ada dalam sebuah startup, tetapi UX tetap menjadi bagian penting dalam pengembangan produk. Startup berteknologi tinggi yang menghadirkan inovasi baru harus memahami bagaimana perasaan pengguna terhadap produk mereka, bahkan lebih dari perusahaan besar. • Proyek dengan anggaran memadai
UX sering diabaikan pada proyek bernilai rendah, tetapi sebuah tim pengembangan dengan anggaran yang cukup cenderung mengalokasikan sebagian sumber dayanya untuk memastikan UX memberikan hasil yang sepadan. • Proyek jangka panjang
Semakin lama durasi sebuah proyek, semakin besar pula sumber daya yang dibutuhkan. Dalam kondisi seperti ini, UX menjadi semakin penting untuk memastikan hasil investasi dapat tercapai. Apa Metodologi Utama dalam UX? Metodologi utama yang digunakan untuk menjamin pengalaman pengguna dalam sebagian besar proyek adalah desain berpusat pada pengguna. Secara sederhana, desain berpusat pada pengguna menekankan perancangan yang didasarkan pada kebutuhan serta perilaku yang diharapkan dari pengguna. Penting bagi desainer pengalaman pengguna untuk memahami bahwa desain berpusat pada pengguna merupakan cara untuk mencapai pengalaman pengguna yang baik, namun bukan satu satunya pendekatan atau alat yang dapat digunakan untuk memastikan pengalaman pengguna yang optimal dalam sebuah proyek. Inti Pembahasan

Desain pengalaman pengguna berperan dalam memandu proses pengembangan produk agar mampu membentuk bagaimana pengguna merasakan produk tersebut saat digunakan. Metode ini tidak sepenuhnya sempurna, karena meski sebuah tim memiliki pengetahuan UX yang matang, suatu produk tetap bisa gagal. Namun, penggunaan prinsip UX secara tepat memberikan peluang yang jauh lebih besar bagi keberhasilan sebuah produk dibandingkan produk yang dirancang tanpa prinsip tersebut. Jika ingin menghindari berbagai kesalahan umum dalam desain UX, Anda dapat mempelajari kursus “User Experience: The Beginner’s Guide” yang menawarkan panduan komprehensif untuk memulai. User Experience: Panduan Pemula
Jika Anda ingin memasuki salah satu bidang desain yang pertumbuhannya paling cepat, maka panduan ini sangat sesuai. Anda akan mempelajari keterampilan yang dibutuhkan untuk membantu perusahaan menghadirkan pengalaman pengguna yang tepat melalui pendekatan yang telah teruji. Anda juga akan mempelajari cara menghasilkan berbagai output UX dengan bantuan templat yang dapat diunduh. Mulai dari peta perjalanan pengguna sampai prototipe berbasis kertas, kursus ini menunjukkan bagaimana memanfaatkan templat tersebut untuk membangun portofolio UX yang kuat, sehingga membantu saat melamar pekerjaan. Selain itu, Anda akan mendapatkan akses ke video wawancara bersama perekrut dan ahli UX dari perusahaan seperti Google dan SAP yang memberikan wawasan mengenai keterampilan dan kualitas yang dicari.
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