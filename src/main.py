# src/main.py
import asyncio
import os
import sys
from dotenv import load_dotenv
import logging
import json

# 確保可以 import 同層或上層資料夾的模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# 以下為原先 managers / adapters / services import
from managers.embedding_manager import EmbeddingManager
from managers.vector_store_manager import VectorStoreManager
from managers.llm_manager import LLMManager
from managers.blacklist_manager import BlacklistManager
from managers.regulations_manager import RegulationsManager

from adapters.openai_adapter import OpenAIAdapter
from adapters.local_llama_adapter import LocalLlamaAdapter

from services.fraud_rag_service import FraudRAGService
from services.compliance_rag_service import ComplianceRAGService, LawComplianceService
from evaluation_utils import save_results_to_json, build_matrix, visualize_matrix

# 載入 .env
load_dotenv(override=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def setup_managers():
    """根據 .env 配置，初始化所有核心 manager。"""

    # 1) 從 .env 讀取環境變數
    openai_api_key = os.getenv("OPENAI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_collection = os.getenv("QDRANT_COLLECTION")
    embedding_model_name = os.getenv("EMBED_MODEL")

    # 2) 建立 EmbeddingManager
    embedding_manager = EmbeddingManager(
        openai_api_key=openai_api_key,
        embedding_model_name=embedding_model_name
    )

    # 3) 建立 VectorStoreManager (Qdrant)
    vector_store_manager = VectorStoreManager(
        embedding_manager=embedding_manager,
        qdrant_url=qdrant_url,
        collection_name=qdrant_collection
    )

    # 4) 建立 LLMManager，並註冊多個 LLM Adapter
    llm_manager = LLMManager()

    openai_adapter = OpenAIAdapter(
        openai_api_key=openai_api_key,
        temperature=0.0,
        max_tokens=1024
    )

    local_llama_adapter = LocalLlamaAdapter(
        model_path="models/llama.bin",
        temperature=0.0,
        max_tokens=2048
    )

    llm_manager.register_adapter("openai", openai_adapter)
    llm_manager.register_adapter("llama", local_llama_adapter)
    llm_manager.set_default_adapter("openai")

    # 4) 其他
    blacklist_manager = BlacklistManager(blacklist_db=["badurl.com", "lineid123"])
    regulations_manager = RegulationsManager(regulations_db={"some_law": "Lorem ipsum..."})

    return embedding_manager, vector_store_manager, llm_manager, blacklist_manager, regulations_manager


def load_jsonl_file(path: str):
    """小幫手：讀取 JSON lines 檔並回傳 list[dict]."""
    if not path or not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return []
    all_data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_data.append(json.loads(line))
    return all_data


async def main():
    # 初始化 Managers
    embedding_manager, vector_store_manager, llm_manager, blacklist_manager, regulations_manager = setup_managers()

    # 1) 讀取詐騙檔 (mock)
    scam_file = os.getenv("SCAM_PATTERNS_FILE")
    scam_patterns = load_jsonl_file(scam_file)

    # 把詐騙patterns 加入 domain="FRAUD" 的向量庫
    vector_store_manager.add_documents(
        domain="FRAUD",
        docs=scam_patterns,
        metadata={"type": "scam_pattern"}
    )

    # 建立 FraudRAGService
    fraud_service = FraudRAGService(
        embedding_manager=embedding_manager,
        vector_store_manager=vector_store_manager,
        llm_manager=llm_manager,
        blacklist_manager=blacklist_manager,
        domain_key="FRAUD",
        selected_llm_name="openai"
    )

    # 2) 測試詐騙
    # user_input_fraud = "這裡有免錢的投資秘笈，而且加Line就能獲得！ lineid123"
    # user_input_fraud = "我於114年04月14日凌晨02時30分在住家瀏覽FACEBOOK社群網路平臺其中社團【嘉義市租屋網】貼文者【鐘仲妍、臉書ID：100086865836189】刊登【租屋廣告】貼文，我遂加入假廣告貼文所提供之聯繫方式【LINE ID：hhh88338】聯繫，俟我上鉤後，對方便要求先預付兩個月訂金才能保留物件並帶看屋況，我遂匯款【14000元】至【對方帳戶:帳戶號碼】，後續對方告知該址已經租出去，然後要退款給我，我提供帳戶給對方進行匯款，但遲遲未見對方匯回，於是我向對方提問，對方即佯稱應該其帳戶出問題，因此又提供我另一line帳號鏈結供我加入詢問，我加入後該line顯示【中 國 信 託】，隨後【中 國 信 託】用line來電，告知我當初匯款的帳戶為問題帳戶，需要透過相關操作才"
    user_input_fraud = "我於114年01月(時間不確定)，不明原因於LINE軟體加入「淑瑤交流技術學院」群組，並在其中認識匿名「林淑瑤」之詐騙嫌疑人，「林淑瑤」以投資股票為由，假借「保勝投資股份有限公司」名義，邀請我投資股票，使用假APP及高獲利方式，使我於114年03月31日17時43分，於我，以新臺幣300000元，與面交車手面交。隨後欲出金卻出不了金，於114年04月15日13時20分，至京城銀行中埔分行(嘉義縣中埔鄉和睦村中山路五段867)，詢問有無資金入帳，經行員面談後，才知悉被詐騙，故至本所報案。"
    # user_input_fraud = "我報稱於114年04月13日下午14時許，我接獲來電(來電號碼已不清楚)，犯嫌以猜猜我是誰方式向我搭話，隨後犯嫌謊稱為我友人「清月」，稱手機掉進水裡無法使用，要求我加入Line暱稱【幸福】之帳號。犯嫌於114年04月14日向我謊稱，渠急需借用15萬元周轉，我不疑有他，於同日10時59分至土地銀行民雄分行臨櫃匯款新臺幣15萬元至對方提供之帳號，直至114年04月15日早上接到真實友人來電查證後，才得知遭詐。"
    result_fraud = await fraud_service.generate_answer(user_input_fraud)
    print("=== Fraud check result ===\n", result_fraud)

    # 3) 讀取外部法規與公司內部規範
    external_law_file = os.getenv("EXTERNAL_LAW_FILE")
    internal_policy_file = os.getenv("INTERNAL_POLICY_FILE")

    external_law_data = load_jsonl_file(external_law_file)     # 外部法規
    internal_policy_data = load_jsonl_file(internal_policy_file)  # 內部規範

    # 4) 新增「外部法規」與「內部規定」到 domain="COMPLIANCE"
    #    差別在 metadata["source"] 設成 "external" 或 "internal"
    #    利用 vector_store_manager.add_jsonl_documents(...) 方便把多欄位存進 metadata
    if external_law_data:
        for item in external_law_data:
            item["source"] = "external"   # 以source區分
        vector_store_manager.add_jsonl_documents(
            domain="COMPLIANCE",
            json_lines=external_law_data,
            text_key="text",
            meta_keys=["source", "doc_name", "chapter_no", "chapter_name", "article_no", "effective_date", "clause_id"]
        )
    if internal_policy_data:
        for item in internal_policy_data:
            item["source"] = "internal"
        vector_store_manager.add_jsonl_documents(
            domain="COMPLIANCE",
            json_lines=internal_policy_data,
            text_key="text",
            meta_keys=["source", "doc_name", "chapter_no", "chapter_name", "article_no", "effective_date", "clause_id"]
        )

    # 5) 建立 ComplianceRAGService + LawComplianceService
    compliance_service = ComplianceRAGService(
        embedding_manager=embedding_manager,
        vector_store_manager=vector_store_manager,
        llm_manager=llm_manager,
        regulations_manager=regulations_manager,
        domain_key="COMPLIANCE",
        selected_llm_name="openai"
    )
    law_service = LawComplianceService(compliance_rag=compliance_service)

    # --- 6) 測試「外部法規 -> 內部規範」對照 ---
    if external_law_data:
        # 先示範只拿前幾條做測試 (減少Token消耗)
        # test_external = external_law_data[:3]
        test_external = external_law_data
        forward_results = await law_service.audit(test_external)

        print("\n=== Forward RAG (external -> internal) 結果 ===")
        print(json.dumps(forward_results, ensure_ascii=False, indent=2))

    # --- (可選) 測試「內部規範 -> 外部法規」對照 ---
    if internal_policy_data:
        test_internal = internal_policy_data
        reverse_results = await law_service.audit_reverse(test_internal)
        print("\n=== Reverse RAG (internal -> external) 結果 ===")
        print(json.dumps(reverse_results, ensure_ascii=False, indent=2))

    # 存檔
    save_results_to_json(forward_results, reverse_results)

    # 建立matrix + 繪圖
    matrix, ext_keys, int_keys = build_matrix(forward_results, reverse_results)
    visualize_matrix(matrix, ext_keys, int_keys, out_png=os.getenv("EVALUATION_IMAGE"))

if __name__ == "__main__":
    asyncio.run(main())
