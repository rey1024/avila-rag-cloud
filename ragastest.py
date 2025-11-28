import numpy as np
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    ContextRecall,
    ContextPrecision
)
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
load_dotenv()
print("✔ Env loaded from .env")

# Data untuk evaluasi
data = {
    'question': [
        "Siapa yang disebut sebagai desainer pengalaman pengguna?"
    ],
    'answer': [
        "Seorang desainer pengalaman pengguna adalah seseorang yang menyelidiki dan menganalisis bagaimana perasaan pengguna terhadap produk yang ia tawarkan.",
    ],
    'contexts': [
        ["""
     [71] Kompleksitas dan Persepsi

Desain pengalaman pengguna berfokus pada bagaimana merancang pengalaman terbaik ketika seseorang menggunakan sebuah layanan atau produk. Bidang ini dapat mencakup berbagai jenis layanan dan produk, termasuk desain pameran di museum. Meskipun cakupannya luas, istilah desain pengalaman pengguna paling sering digunakan dalam konteks situs web, aplikasi berbasis web, dan aplikasi perangkat lunak. Sejak paruh kedua dekade pertama abad ini, teknologi berkembang semakin kompleks, dan fungsi aplikasi maupun situs web menjadi jauh lebih luas dan rumit. Pada awal kemunculannya, situs web hanyalah halaman statis sederhana yang menyajikan informasi bagi pengguna yang ingin mencari sesuatu. Namun, beberapa dekade kemudian, kita menemukan berbagai situs yang interaktif dan mampu menghadirkan pengalaman yang jauh lebih kaya bagi para penggunanya. Anda dapat menambahkan berbagai fitur dan fungsi apa pun ke sebuah situs atau aplikasi, tetapi keberhasilan sebuah proyek tetap bergantung pada satu faktor utama: bagaimana perasaan pengguna terhadapnya. “Manusia selalu bersifat emosional dan selalu bereaksi secara emosional terhadap artefak dalam dunia mereka.”
—Alan Cooper, President of Cooper
Pertanyaan yang menjadi perhatian para desainer pengalaman pengguna meliputi hal hal berikut:
• Apakah situs atau aplikasi tersebut memberikan nilai bagi pengguna? • Apakah pengguna merasa situs atau aplikasi tersebut mudah digunakan dan mudah dinavigasi? • Apakah pengguna benar benar menikmati pengalaman menggunakan situs atau aplikasi tersebut? Seorang desainer pengalaman pengguna dapat mengatakan bahwa ia telah melakukan pekerjaannya dengan baik ketika semua pertanyaan tersebut dapat dijawab dengan “Ya.”
Apa itu Pengalaman Pengguna (UX)? Secara umum, pengalaman pengguna menggambarkan bagaimana perasaan seseorang ketika menggunakan sebuah produk atau layanan. Dalam banyak kasus, produk tersebut berupa situs web atau aplikasi. Setiap bentuk interaksi antara manusia dan objek selalu memiliki pengalaman pengguna terkait, namun pada umumnya para praktisi UX berfokus pada hubungan antara pengguna manusia dan komputer serta produk berbasis komputer seperti situs web, aplikasi, dan sistem. Apa itu Desainer UX? Seorang desainer pengalaman pengguna adalah seseorang yang menyelidiki dan menganalisis bagaimana perasaan pengguna terhadap produk yang ia tawarkan. Desainer UX kemudian menerapkan wawasan tersebut dalam proses pengembangan produk untuk memastikan pengguna memperoleh pengalaman terbaik saat menggunakan produk tersebut. Desainer UX melakukan berbagai kegiatan, seperti melakukan riset, menganalisis temuan, menyampaikan hasil temuan kepada anggota tim pengembangan, memantau jalannya proyek untuk memastikan temuan tersebut diterapkan, dan banyak tugas lainnya. Mengapa UX Penting? Dulu, desain produk cenderung sederhana. Para desainer membuat sesuatu yang menurut mereka keren dan berharap klien menyukainya. Namun, ada dua masalah utama dengan pendekatan tersebut. Pertama, persaingan untuk mendapatkan perhatian pengguna jauh lebih sedikit pada masa itu. Kedua, tidak ada pertimbangan terhadap pengguna produk sama sekali. Keberhasilan atau kegagalan sebuah proyek desain lebih banyak ditentukan oleh keberuntungan dan penilaian subjektif tim desain. Fokus pada pengalaman pengguna membuat proses desain lebih berorientasi pada kebutuhan manusia.
[90] Teknik teknik ini berasal dari bidang tradisional maupun lintas disiplin, serta berkembang seiring kemajuan teknologi yang mendukung praktik desain. Seiring meningkatnya kecanggihan metode dan alat riset, biayanya justru semakin terjangkau. Karena itu, tidak ada alasan untuk tidak menggunakan kombinasi teknik tersebut demi menghasilkan desain terbaik. Anda kini memiliki tujuh teknik riset UX tambahan dalam perangkat kerja Anda. Tetapi, teknik mana yang sebaiknya digunakan dalam proyek Anda? Kapan card sorting efektif, dan kapan justru tidak produktif? Untuk membantu Anda memahami berbagai cara melakukan riset pengguna secara tepat, tersedia kursus “User Research - Methods and Best Practices” yang berisi wawasan praktis sehingga Anda tidak lagi menjadi pemula dalam riset pengguna. What is Interaction Design? Interaction design merupakan komponen penting dalam payung besar desain pengalaman pengguna. Pada bagian ini dijelaskan apa itu interaction design, beberapa model interaction design yang berguna, serta gambaran singkat mengenai tugas seorang interaction designer. Pemahaman Sederhana dan Berguna tentang Interaction Design

Interaction design dapat dipahami dalam istilah yang sederhana namun tetap akurat: ini adalah proses merancang interaksi antara pengguna dan produk. Dalam banyak kasus, produk tersebut berupa perangkat lunak seperti aplikasi atau situs web. Tujuan interaction design adalah menciptakan produk yang memungkinkan pengguna mencapai tujuan mereka dengan cara yang paling optimal. Jika definisi ini terdengar luas, itu karena bidangnya memang luas. Interaksi antara pengguna dan produk melibatkan berbagai elemen seperti estetika, gerakan, suara, ruang, dan banyak lagi. Setiap elemen tersebut bahkan memiliki bidang keahlian khusus—misalnya, sound design untuk merancang suara yang digunakan dalam interaksi pengguna. Sebagaimana terlihat, terdapat tumpang tindih besar antara interaction design dan UX design. UX design berfokus pada pembentukan pengalaman menggunakan produk, dan pengalaman tersebut hampir selalu melibatkan interaksi antara pengguna dan produk. Namun, UX design mencakup hal yang lebih luas daripada interaction design. UX juga melibatkan riset pengguna (mengetahui siapa penggunanya), pembuatan user persona (mengapa dan dalam kondisi apa mereka menggunakan produk), melakukan user testing dan usability testing, dan masih banyak lagi. “Saat menciptakan konten, tunjukkan empati di atas segalanya. Cobalah menjalani kehidupan audiens Anda.”
— Rand Fishkin, Founder Moz
Lima Dimensi Interaction Design
Model lima dimensi interaction design membantu memahami cakupan dan ruang lingkup kerja seorang interaction designer. Konsep ini pertama kali diperkenalkan oleh Gillian Crampton Smith yang mendefinisikan empat dimensi bahasa desain interaksi, kemudian dikembangkan oleh Kevin Silver dari IDEXX Laboratories dengan menambahkan dimensi kelima. 1D: Words
Words mencakup kata kata yang digunakan dalam interaksi, misalnya label tombol. Kata harus bermakna dan mudah dipahami, menyampaikan informasi secara tepat tanpa berlebihan, agar tidak membingungkan atau memperlambat pengguna. 2D: Visual Representations
Dimensi ini meliputi elemen grafis seperti gambar, tipografi, dan ikon yang digunakan pengguna untuk berinteraksi. Elemen visual biasanya melengkapi kata kata dan menjadi bagian penting bagi pengalaman pengguna.
[77] Menghasilkan Ide dan Solusi Kreatif melalui Pemahaman Manusia secara Holistik
Dengan landasan yang kuat pada sains dan rasionalitas, design thinking berupaya membangun pemahaman holistik dan empatik terhadap masalah yang dihadapi manusia. Pendekatan ini mencoba memahami manusia secara mendalam, termasuk berbagai konsep yang bersifat ambigu atau subjektif seperti emosi, kebutuhan, motivasi, dan pendorong perilaku. Karakter proses menghasilkan ide dan solusi dalam design thinking membuat pendekatan ini lebih peka dan lebih tertarik pada konteks tempat pengguna beroperasi serta masalah atau hambatan yang mereka hadapi ketika berinteraksi dengan sebuah produk. Unsur kreatif dalam design thinking muncul dari metode yang digunakan untuk menghasilkan solusi dan memperoleh wawasan mengenai praktik, tindakan, serta pola pikir pengguna nyata. Design Thinking adalah Proses Iteratif dan Tidak Linear
Design thinking merupakan proses yang iteratif dan tidak linear. Artinya, tim desain terus menerus menggunakan hasil yang mereka peroleh untuk meninjau, mempertanyakan, dan meningkatkan asumsi awal, pemahaman, serta hasil sebelumnya. Temuan dari tahap akhir proses awal akan memperkaya pemahaman kita tentang masalah, membantu menentukan parameter masalah, memungkinkan kita mendefinisikan ulang masalah, dan yang paling penting memberikan wawasan baru sehingga kita dapat melihat berbagai solusi alternatif yang sebelumnya tidak terlihat pada tingkat pemahaman awal. Design Thinking: Panduan Pemula

Perusahaan perusahaan terkemuka dunia seperti Apple, Google, dan Samsung telah menggunakan pendekatan design thinking karena mereka memahami bahwa pendekatan ini menjadi jalan utama menuju inovasi dan keberhasilan produk. Melalui Design Thinking: The Beginner’s Guide, Anda akan mempelajari secara mendalam lima fase pendekatan pemecahan masalah yang mengubah paradigma ini, yaitu empathize, define, ideate, prototype, dan test. Dengan panduan rinci mengenai berbagai aktivitas pemecahan masalah, mulai dari teknik menghasilkan ide seperti brainstorming dan penggunaan analogi, hingga cara mengumpulkan umpan balik dari prototipe, Anda dapat mengunduh berbagai templat yang tersedia dan menggunakannya secara efektif dalam pekerjaan Anda. Bersiaplah untuk mempelajari, mengeksplorasi, dan menguasai design thinking sehingga dapat menjadi nilai tambah dan membuka tahap berikutnya dalam perjalanan profesional Anda. Tujuh Faktor yang Mempengaruhi Pengalaman Pengguna

Pengalaman pengguna memiliki peran penting dalam menentukan keberhasilan atau kegagalan sebuah produk di pasar. Namun, apa sebenarnya yang dimaksud dengan pengalaman pengguna? Banyak orang sering menyamakan pengalaman pengguna dengan usability, yaitu seberapa mudah suatu produk digunakan. Meskipun benar bahwa bidang pengalaman pengguna berawal dari konsep usability, cakupannya kini jauh lebih luas. Memperhatikan seluruh aspek pengalaman pengguna menjadi hal penting agar sebuah produk dapat berhasil di pasar. “Untuk menjadi desainer yang hebat, Anda perlu memahami lebih dalam bagaimana orang berpikir dan bertindak.”
— Paul Boag, Co-Founder Headscape Limited
Peter Morville, salah satu pelopor dalam bidang pengalaman pengguna yang menulis berbagai buku terlaris dan menjadi penasihat bagi banyak perusahaan Fortune 500, menyusun tujuh faktor yang menggambarkan pengalaman pengguna.
        """]
    ],
    'ground_truth': [
        """
       Seorang desainer pengalaman pengguna adalah seseorang yang menyelidiki dan menganalisis bagaimana perasaan pengguna terhadap produk yang ia tawarkan.
        """
    ]
}

dataset = Dataset.from_dict(data)
openai_key=os.getenv("OPENAI_API_KEY")
# -------------------------------------------------
# Evaluasi RAGAS
# -------------------------------------------------
ragas_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=openai_key
)
ragas_embed = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=openai_key
)
ragas_embed.client.batch_size = 128

result = evaluate(
    dataset=dataset,
    metrics=[
        ContextPrecision(),
        AnswerRelevancy(),
        Faithfulness(),
        ContextRecall()
    ],
    llm=ragas_llm,
    embeddings=ragas_embed
)

# Helper untuk meng-handle float atau Score.value
def to_float(x):
    if hasattr(x, "value"):
        return float(x.value)
    return float(x)

# -------------------------------------------------
# Cetak nilai per item
# -------------------------------------------------
print("=" * 50)
print("HASIL EVALUASI RAGAS MULTI SAMPLE")
print("=" * 50)

n = len(data["question"])
for i in range(n):
    print(f"\nItem {i+1}")
    print(f"  context_precision : {to_float(result['context_precision'][i]):.3f}")
    print(f"  answer_relevancy  : {to_float(result['answer_relevancy'][i]):.3f}")
    print(f"  faithfulness      : {to_float(result['faithfulness'][i]):.3f}")
    print(f"  context_recall    : {to_float(result['context_recall'][i]):.3f}")

# -------------------------------------------------
# Aggregate score
# -------------------------------------------------
agg = {
    "context_precision": np.mean([to_float(x) for x in result["context_precision"]]),
    "answer_relevancy": np.mean([to_float(x) for x in result["answer_relevancy"]]),
    "faithfulness": np.mean([to_float(x) for x in result["faithfulness"]]),
    "context_recall": np.mean([to_float(x) for x in result["context_recall"]])
}

print("\n==============================")
print("AGGREGATE SCORE")
print("==============================")
for k, v in agg.items():
    print(f"{k:20s} {v:.4f}")

print("\nANALISIS KUALITAS")
for k, v in agg.items():
    if v >= 0.8:
        status = "EXCELLENT"
    elif v >= 0.6:
        status = "GOOD"
    elif v >= 0.4:
        status = "FAIR"
    else:
        status = "POOR"
    print(f"{k:20s} {v:.4f}  {status}")
