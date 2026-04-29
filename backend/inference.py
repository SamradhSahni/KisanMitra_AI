import os
import sys
import torch
import json
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

# ── Config ────────────────────────────────────────────────────────────
FINAL_MODEL_DIR = os.getenv("FINETUNED_MODEL_PATH", "./model/final")
MAX_INPUT_LEN   = int(os.getenv("MAX_INPUT_LENGTH",  512))
MAX_NEW_TOKENS  = int(os.getenv("MAX_OUTPUT_LENGTH", 128))

# ── Hindi display maps (same as format_dataset.py) ───────────────────
CROP_HINDI_MAP = {
    "others":                                            "अन्य",
    "paddy (dhan)":                                      "धान",
    "wheat":                                             "गेहूं",
    "maize (makka)":                                     "मक्का",
    "green gram (moong bean/ moong)":                    "मूंग",
    "pearl millet (bajra/bulrush millet/spiked millet)": "बाजरा",
    "mustard":                                           "सरसों",
    "sugarcane (noble cane)":                            "गन्ना",
    "cotton (kapas)":                                    "कपास",
    "groundnut (pea nut/mung phalli)":                   "मूंगफली",
    "mango":                                             "आम",
    "potato":                                            "आलू",
    "onion":                                             "प्याज",
    "bengal gram (gram/chick pea/kabuli/chana)":         "चना",
    "tomato":                                            "टमाटर",
    "guar":                                              "ग्वार",
    "soybean (bhat)":                                    "सोयाबीन",
    "cumin":                                             "जीरा",
    "black gram (urd bean)":                             "उड़द",
    "chillies":                                          "मिर्च",
    "barley":                                            "जौ",
    "lentil (masoor)":                                   "मसूर",
    "arhar (tur/red gram/pigeon pea)":                   "अरहर",
    "coriander":                                         "धनिया",
    "garlic":                                            "लहसुन",
    "sunflower":                                         "सूरजमुखी",
    "sesame (til)":                                      "तिल",
    "isabgol":                                           "ईसबगोल",
    "bitter gourd":                                      "करेला",
    "bottle gourd":                                      "लौकी",
    "brinjal":                                           "बैंगन",
    "cabbage":                                           "पत्तागोभी",
    "cauliflower":                                       "फूलगोभी",
    "cucumber":                                          "खीरा",
    "pea":                                               "मटर",
    "spinach":                                           "पालक",
    "radish":                                            "मूली",
    "carrot":                                            "गाजर",
    "capsicum":                                          "शिमला मिर्च",
    "papaya":                                            "पपीता",
    "banana":                                            "केला",
    "guava":                                             "अमरूद",
    "pomegranate":                                       "अनार",
    "apple":                                             "सेब",
    "grape":                                             "अंगूर",
    "turmeric":                                          "हल्दी",
    "ginger":                                            "अदरक",
    "coconut":                                           "नारियल",
}

INTENT_HINDI_MAP = {
    "weather_sowing":      "मौसम एवं बुवाई",
    "crop_advisory":       "फसल सलाह",
    "pest_id":             "कीट प्रबंधन",
    "disease":             "रोग प्रबंधन",
    "nutrient_management": "पोषक तत्व प्रबंधन",
    "msp_price":           "मूल्य एवं बाजार",
    "government_scheme":   "सरकारी योजना",
    "horticulture":        "बागवानी",
    "soil_water":          "मृदा एवं जल प्रबंधन",
    "animal_husbandry":    "पशुपालन",
    "equipment_machinery": "कृषि यंत्र",
    "unknown":             "सामान्य",
}

STATE_HINDI_MAP = {
    "UTTAR PRADESH":    "उत्तर प्रदेश",
    "RAJASTHAN":        "राजस्थान",
    "MADHYA PRADESH":   "मध्य प्रदेश",
    "BIHAR":            "बिहार",
    "HARYANA":          "हरियाणा",
    "JHARKHAND":        "झारखंड",
    "UTTARAKHAND":      "उत्तराखंड",
    "CHHATTISGARH":     "छत्तीसगढ़",
    "HIMACHAL PRADESH": "हिमाचल प्रदेश",
    "DELHI":            "दिल्ली",
}


# ── Intent detector (query-time) ──────────────────────────────────────
def detect_intent(query: str) -> str:
    """
    Detect intent from a Hindi query at inference time.
    Uses the same keyword banks as training pipeline.
    """
    from utils.config_loader import CONFIG
    intents = CONFIG["intents"]
    query_lower = query.lower()

    for intent_name, kw_banks in intents.items():
        for lang, words in kw_banks.items():
            for word in words:
                if word.lower() in query_lower:
                    return intent_name
    return "unknown"


# ── Build instruction prompt ──────────────────────────────────────────
def build_prompt(
    query:  str,
    state:  str  = "UTTAR PRADESH",
    crop:   str  = "others",
    intent: str  = None,
    rag_context: str = None,
) -> str:
    """
    Build the full mT5 instruction prompt.
    Optionally prepend RAG context passages (used in Task 12+).
    """
    # Auto-detect intent if not provided
    if intent is None:
        intent = detect_intent(query)

    state_hi  = STATE_HINDI_MAP.get(state.upper(), state)
    crop_hi   = CROP_HINDI_MAP.get(crop.lower().strip(), crop)
    intent_hi = INTENT_HINDI_MAP.get(intent, "सामान्य")

    # RAG context block
    context_block = ""
    if rag_context:
        context_block = (
            f"संदर्भ जानकारी:\n{rag_context}\n"
        )

    prompt = (
        f"निर्देश: आप एक कृषि विशेषज्ञ हैं। किसान की समस्या का उत्तर हिंदी में दें।\n"
        f"राज्य: {state_hi}\n"
        f"फसल: {crop_hi}\n"
        f"समस्या का प्रकार: {intent_hi}\n"
        f"{context_block}"
        f"किसान का प्रश्न: {query}\n"
        f"उत्तर:"
    )
    return prompt


# ── KisanMitraInference class ─────────────────────────────────────────
class KisanMitraInference:
    """
    Standalone inference class for the fine-tuned mT5 model.
    Loaded once at startup — reused for all requests.

    Usage:
        engine = KisanMitraInference()
        engine.load()
        response = engine.generate(
            query="गेहूं में कीट नियंत्रण कैसे करें?",
            state="HARYANA",
            crop="wheat",
        )
    """

    def __init__(self, model_dir: str = FINAL_MODEL_DIR):
        self.model_dir  = model_dir
        self.model      = None
        self.tokenizer  = None
        self.device     = None
        self._loaded    = False

    def load(self):
        """Load model and tokenizer into memory. Call once at startup."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self._loaded:
            logger.info("Model already loaded — skipping")
            return

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Loading KisanMitra model from: {self.model_dir}")
        logger.info(f"Device: {self.device}")

        # Verify model directory exists
        model_path = Path(self.model_dir)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at: {self.model_dir}\n"
                "Run save_final_model.py first (Task 8b)."
            )

        required_files = ["config.json", "tokenizer_config.json"]
        for f in required_files:
            if not (model_path / f).exists():
                raise FileNotFoundError(
                    f"Missing {f} in model directory. "
                    "Model may not have saved correctly."
                )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            use_fast=True,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.model.eval()
        self._loaded = True

        # Log memory usage
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1024**3
            logger.success(f"Model loaded — GPU memory used: {mem_used:.2f} GB")
        logger.success("KisanMitraInference ready")

    def generate(
        self,
        query:       str,
        state:       str  = "UTTAR PRADESH",
        crop:        str  = "others",
        intent:      str  = None,
        rag_context: str  = None,
        num_beams:   int  = 4,
        max_new_tokens: int = None,
    ) -> dict:
        """
        Generate a Hindi advisory response.

        Args:
            query        : Hindi farmer question
            state        : farmer's state (English uppercase)
            crop         : crop name (English lowercase)
            intent       : override intent detection (optional)
            rag_context  : RAG-retrieved passages (optional, Task 12)
            num_beams    : beam search width (4 = quality, 1 = speed)
            max_new_tokens: override max output length

        Returns:
            dict with keys: response, intent, prompt, tokens_generated
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call engine.load() first.")

        if max_new_tokens is None:
            max_new_tokens = MAX_NEW_TOKENS

        # Detect intent
        detected_intent = detect_intent(query) if intent is None else intent

        # Build prompt
        prompt = build_prompt(
            query=query,
            state=state,
            crop=crop,
            intent=detected_intent,
            rag_context=rag_context,
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding=True,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
            )

        # Decode
        response = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        tokens_generated = output_ids.shape[-1]

        return {
            "response":         response,
            "intent":           detected_intent,
            "prompt":           prompt,
            "tokens_generated": tokens_generated,
            "model_dir":        self.model_dir,
        }

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self):
        """Free GPU memory if needed."""
        if self._loaded:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._loaded = False
            logger.info("Model unloaded from memory")


# ── Module-level singleton ────────────────────────────────────────────
# Import and use this in FastAPI backend
_engine: KisanMitraInference = None


def get_engine() -> KisanMitraInference:
    """
    Return the singleton inference engine.
    Loads on first call, reuses on subsequent calls.
    """
    global _engine
    if _engine is None:
        _engine = KisanMitraInference()
        _engine.load()
    return _engine