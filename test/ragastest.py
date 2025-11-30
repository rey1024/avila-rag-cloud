import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

from ragas.metrics.collections import (
    ContextPrecision,
    AnswerRelevancy,
    Faithfulness,
    AnswerCorrectness     # <= DITAMBAHKAN
)

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import LLMContextRecall
import logging
logging.basicConfig(level=logging.INFO)

async def main():

    # -----------------------------------------
    # 1. Setup LLM + embeddings
    # -----------------------------------------

    client_llm = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    client_embed = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    client_llm_ragas = llm_factory(
        model="gpt-4o-mini",
        client=client_llm
    )

    embeddings = embedding_factory(
        provider="openai",
        model="text-embedding-3-small",
        client=client_embed
    )

    # -----------------------------------------
    # 2. Dataset (isi sesuai kebutuhan)
    # -----------------------------------------
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

    # -----------------------------------------
    # 3. Siapkan metrik
    # -----------------------------------------
    m_cp = ContextPrecision(llm=client_llm_ragas)
    m_f  = Faithfulness(llm=client_llm_ragas)
    m_ar = AnswerRelevancy(llm=client_llm_ragas, embeddings=embeddings)
    m_ac = AnswerCorrectness(llm=client_llm_ragas, embeddings=embeddings)   # <= DITAMBAHKAN

    from langchain_openai import ChatOpenAI
    evaluator_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    m_cr = LLMContextRecall(llm=evaluator_llm)

    # -----------------------------------------
    # 4. Evaluasi per item
    # -----------------------------------------
    for i in range(n):
        q   = ds["question"][i]
        r   = ds["answer"][i]
        gt  = ds["ground_truth"][i]
        ctx = ds["contexts"][i]

        # --- Context Precision ---
        CP = await m_cp.ascore(
            user_input=q,
            reference=gt,
            retrieved_contexts=ctx
        )

        # --- Faithfulness ---
        F = await m_f.ascore(
            user_input=q,
            response=r,
            retrieved_contexts=ctx
        )

        # --- Answer Relevancy ---
        AR = await m_ar.ascore(
            user_input=q,
            response=r
        )

        # --- Answer Correctness ---
        AC = await m_ac.ascore(
            user_input=q,
            response=r,
            reference=gt
        )

        # --- Context Recall ---
        sample = SingleTurnSample(
            user_input=q,
            response=r,
            reference=gt,
            retrieved_contexts=ctx
        )
        CR = await m_cr.single_turn_ascore(sample)

        print("=" * 50)
        print(f"Item {i+1}")
        print(f"Context Precision : {float(CP.value):.4f}")
        print(f"Context Recall    : {float(CR):.4f}")
        print(f"Faithfulness      : {float(F.value):.4f}")
        print(f"Answer Relevancy  : {float(AR.value):.4f}")
        print(f"Answer Correctness: {float(AC.value):.4f}")   # <= DITAMPILKAN

if __name__ == "__main__":
    asyncio.run(main())
