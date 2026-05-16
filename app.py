import torch
import gradio as gr
from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
)


# =========================================================
# 1. 장치 설정
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("사용 장치:", device)


# =========================================================
# 2. 오픈소스 BLIP 모델 로드
#    역할: 이미지 캡션 생성
# =========================================================
print("BLIP 모델 로딩 중...")
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)
print("BLIP 모델 로딩 완료")


# =========================================================
# 3. 오픈소스 CLIP 모델 로드
#    역할: 이미지와 음식 후보 텍스트 유사도 비교
# =========================================================
print("CLIP 모델 로딩 중...")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
print("CLIP 모델 로딩 완료")


# =========================================================
# 4. CLIP 출력값 처리 함수
# =========================================================
def extract_tensor(output):
    if isinstance(output, torch.Tensor):
        return output

    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output

    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds

    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds

    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]

    raise TypeError("CLIP 출력값에서 Tensor를 추출할 수 없습니다.")


def normalize_features(features):
    return features / features.norm(dim=-1, keepdim=True)


# =========================================================
# 5. 전세계 음식 후보 리스트
#    이 리스트가 많아질수록 대응 가능한 음식 범위가 넓어진다.
# =========================================================
FOOD_CANDIDATES = [
    # Korean
    {"ko": "김치찌개", "en": "kimchi stew", "category": "stew"},
    {"ko": "된장찌개", "en": "doenjang soybean paste stew", "category": "stew"},
    {"ko": "순두부찌개", "en": "soft tofu stew", "category": "stew"},
    {"ko": "부대찌개", "en": "budae jjigae army stew", "category": "stew"},
    {"ko": "갈비탕", "en": "galbitang beef short rib soup", "category": "soup"},
    {"ko": "삼계탕", "en": "samgyetang ginseng chicken soup", "category": "soup"},
    {"ko": "비빔밥", "en": "bibimbap mixed rice bowl", "category": "rice"},
    {"ko": "볶음밥", "en": "fried rice", "category": "rice"},
    {"ko": "김밥", "en": "gimbap seaweed rice roll", "category": "rice"},
    {"ko": "떡볶이", "en": "tteokbokki spicy rice cakes", "category": "snack"},
    {"ko": "불고기", "en": "bulgogi grilled marinated beef", "category": "meat"},
    {"ko": "삼겹살", "en": "grilled pork belly", "category": "meat"},
    {"ko": "제육볶음", "en": "spicy stir fried pork", "category": "meat"},
    {"ko": "잡채", "en": "japchae glass noodles", "category": "noodle"},
    {"ko": "냉면", "en": "cold buckwheat noodles", "category": "noodle"},
    {"ko": "라면", "en": "ramen instant noodles", "category": "noodle"},
    {"ko": "칼국수", "en": "knife cut noodle soup", "category": "noodle"},
    {"ko": "파전", "en": "korean scallion pancake", "category": "pancake"},
    {"ko": "치킨", "en": "fried chicken", "category": "fried"},
    {"ko": "족발", "en": "braised pig trotters", "category": "meat"},

    # Japanese
    {"ko": "스시", "en": "sushi", "category": "rice"},
    {"ko": "사시미", "en": "sashimi raw fish", "category": "fish"},
    {"ko": "라멘", "en": "japanese ramen", "category": "noodle"},
    {"ko": "우동", "en": "udon noodle soup", "category": "noodle"},
    {"ko": "소바", "en": "soba noodles", "category": "noodle"},
    {"ko": "돈카츠", "en": "tonkatsu pork cutlet", "category": "fried"},
    {"ko": "규동", "en": "gyudon beef rice bowl", "category": "rice"},
    {"ko": "카레라이스", "en": "japanese curry rice", "category": "rice"},
    {"ko": "오코노미야키", "en": "okonomiyaki japanese savory pancake", "category": "pancake"},
    {"ko": "타코야키", "en": "takoyaki octopus balls", "category": "snack"},

    # Chinese
    {"ko": "마파두부", "en": "mapo tofu", "category": "stir_fry"},
    {"ko": "짜장면", "en": "jajangmyeon black bean noodles", "category": "noodle"},
    {"ko": "짬뽕", "en": "spicy seafood noodle soup", "category": "noodle"},
    {"ko": "탕수육", "en": "sweet and sour pork", "category": "fried"},
    {"ko": "딤섬", "en": "dim sum dumplings", "category": "dumpling"},
    {"ko": "샤오롱바오", "en": "xiaolongbao soup dumplings", "category": "dumpling"},
    {"ko": "훠궈", "en": "hot pot", "category": "stew"},
    {"ko": "볶음면", "en": "chow mein stir fried noodles", "category": "noodle"},
    {"ko": "북경오리", "en": "peking duck", "category": "meat"},

    # Italian / Western
    {"ko": "토마토 파스타", "en": "tomato pasta spaghetti", "category": "pasta"},
    {"ko": "크림 파스타", "en": "cream pasta", "category": "pasta"},
    {"ko": "알리오 올리오", "en": "aglio e olio pasta", "category": "pasta"},
    {"ko": "라자냐", "en": "lasagna", "category": "pasta"},
    {"ko": "피자", "en": "pizza", "category": "bread"},
    {"ko": "리조또", "en": "risotto", "category": "rice"},
    {"ko": "스테이크", "en": "steak", "category": "meat"},
    {"ko": "햄버거", "en": "hamburger cheeseburger", "category": "bread"},
    {"ko": "샌드위치", "en": "sandwich", "category": "bread"},
    {"ko": "샐러드", "en": "salad", "category": "salad"},
    {"ko": "수프", "en": "soup", "category": "soup"},
    {"ko": "오믈렛", "en": "omelette", "category": "egg"},
    {"ko": "프렌치토스트", "en": "french toast", "category": "bread"},

    # Mexican / Latin
    {"ko": "타코", "en": "taco", "category": "wrap"},
    {"ko": "부리또", "en": "burrito", "category": "wrap"},
    {"ko": "퀘사디아", "en": "quesadilla", "category": "wrap"},
    {"ko": "나초", "en": "nachos", "category": "snack"},
    {"ko": "엔칠라다", "en": "enchilada", "category": "wrap"},
    {"ko": "과카몰리", "en": "guacamole", "category": "salad"},
    {"ko": "세비체", "en": "ceviche", "category": "fish"},

    # Indian / Middle Eastern
    {"ko": "치킨 커리", "en": "chicken curry", "category": "stew"},
    {"ko": "버터 치킨", "en": "butter chicken curry", "category": "stew"},
    {"ko": "난", "en": "naan bread", "category": "bread"},
    {"ko": "비리야니", "en": "biryani rice", "category": "rice"},
    {"ko": "탄두리 치킨", "en": "tandoori chicken", "category": "meat"},
    {"ko": "사모사", "en": "samosa", "category": "fried"},
    {"ko": "케밥", "en": "kebab grilled meat", "category": "meat"},
    {"ko": "팔라펠", "en": "falafel", "category": "fried"},
    {"ko": "후무스", "en": "hummus", "category": "dip"},
    {"ko": "샥슈카", "en": "shakshuka eggs tomato sauce", "category": "egg"},

    # Southeast Asian
    {"ko": "쌀국수", "en": "pho vietnamese noodle soup", "category": "noodle"},
    {"ko": "반미", "en": "banh mi sandwich", "category": "bread"},
    {"ko": "분짜", "en": "bun cha vietnamese noodles", "category": "noodle"},
    {"ko": "팟타이", "en": "pad thai noodles", "category": "noodle"},
    {"ko": "똠얌꿍", "en": "tom yum soup", "category": "soup"},
    {"ko": "그린커리", "en": "thai green curry", "category": "stew"},
    {"ko": "나시고렝", "en": "nasi goreng fried rice", "category": "rice"},
    {"ko": "미고렝", "en": "mie goreng fried noodles", "category": "noodle"},
    {"ko": "사테", "en": "satay skewers", "category": "meat"},

    # Dessert / bakery
    {"ko": "케이크", "en": "cake", "category": "dessert"},
    {"ko": "치즈케이크", "en": "cheesecake", "category": "dessert"},
    {"ko": "초콜릿 케이크", "en": "chocolate cake", "category": "dessert"},
    {"ko": "쿠키", "en": "cookies", "category": "dessert"},
    {"ko": "브라우니", "en": "brownie", "category": "dessert"},
    {"ko": "도넛", "en": "donut", "category": "dessert"},
    {"ko": "크루아상", "en": "croissant", "category": "bread"},
    {"ko": "와플", "en": "waffle", "category": "dessert"},
    {"ko": "팬케이크", "en": "pancakes", "category": "pancake"},
    {"ko": "아이스크림", "en": "ice cream", "category": "dessert"},
]


# =========================================================
# 6. CLIP 텍스트 후보 임베딩 사전 계산
# =========================================================
print("CLIP 음식 후보 임베딩 생성 중...")

TEXT_PROMPTS = []
PROMPT_TO_FOOD_INDEX = []

for idx, food in enumerate(FOOD_CANDIDATES):
    en = food["en"]

    prompts = [
        f"a photo of {en}",
        f"a bowl of {en}",
        f"a plate of {en}",
        f"{en}, food",
    ]

    for p in prompts:
        TEXT_PROMPTS.append(p)
        PROMPT_TO_FOOD_INDEX.append(idx)

with torch.no_grad():
    text_inputs = clip_processor(
        text=TEXT_PROMPTS,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    raw_text_features = clip_model.get_text_features(**text_inputs)
    text_features = extract_tensor(raw_text_features)
    text_features = normalize_features(text_features)

print("CLIP 음식 후보 임베딩 생성 완료")


# =========================================================
# 7. BLIP 캡션 생성
# =========================================================
def generate_blip_caption(image):
    image = image.convert("RGB")

    captions = []

    with torch.no_grad():
        inputs = blip_processor(
            image,
            return_tensors="pt",
        ).to(device)

        output = blip_model.generate(
            **inputs,
            max_new_tokens=50,
        )

        caption = blip_processor.decode(
            output[0],
            skip_special_tokens=True,
        )
        captions.append(caption)

        inputs_food = blip_processor(
            image,
            text="a photo of food:",
            return_tensors="pt",
        ).to(device)

        output_food = blip_model.generate(
            **inputs_food,
            max_new_tokens=50,
        )

        caption_food = blip_processor.decode(
            output_food[0],
            skip_special_tokens=True,
        )
        captions.append(caption_food)

    captions = [c.strip() for c in captions if c and c.strip()]
    captions = sorted(captions, key=len, reverse=True)

    if captions:
        return captions[0]

    return "food"


# =========================================================
# 8. CLIP 기반 음식 Top-K 추정
# =========================================================
def predict_food_with_clip(image, top_k=5):
    image = image.convert("RGB")

    with torch.no_grad():
        image_inputs = clip_processor(
            images=image,
            return_tensors="pt",
        ).to(device)

        raw_image_features = clip_model.get_image_features(**image_inputs)
        image_features = extract_tensor(raw_image_features)
        image_features = normalize_features(image_features)

        similarities = (image_features @ text_features.T).squeeze(0)

    food_scores = {}

    for prompt_idx, score in enumerate(similarities.tolist()):
        food_idx = PROMPT_TO_FOOD_INDEX[prompt_idx]

        if food_idx not in food_scores:
            food_scores[food_idx] = score
        else:
            food_scores[food_idx] = max(food_scores[food_idx], score)

    sorted_foods = sorted(
        food_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []

    for food_idx, score in sorted_foods[:top_k]:
        food = FOOD_CANDIDATES[food_idx]
        results.append({
            "ko": food["ko"],
            "en": food["en"],
            "category": food["category"],
            "score": score,
        })

    return results


# =========================================================
# 9. 한국어 레시피 생성
# =========================================================
def generate_recipe(food, user_request):
    food_name = food["ko"]
    category = food["category"]

    base_ingredients = {
        "stew": ["주재료", "양파", "대파", "다진 마늘", "고춧가루 또는 양념", "물 또는 육수"],
        "soup": ["주재료", "대파", "다진 마늘", "소금", "후추", "물 또는 육수"],
        "rice": ["밥", "주재료", "채소", "간장 또는 소스", "참기름 또는 기름"],
        "noodle": ["면", "주재료", "채소", "소스 또는 육수", "고명"],
        "pasta": ["파스타면", "소스", "마늘", "양파", "올리브유", "치즈 또는 허브"],
        "meat": ["고기", "소금", "후추", "마늘", "양파", "소스 또는 양념"],
        "fish": ["생선", "소금", "후추", "레몬 또는 소스", "채소"],
        "fried": ["주재료", "튀김가루 또는 빵가루", "계란", "식용유", "소스"],
        "bread": ["빵 또는 도우", "주재료", "소스", "치즈 또는 채소"],
        "wrap": ["또르티야 또는 랩", "고기 또는 채소", "소스", "치즈", "토핑"],
        "salad": ["채소", "단백질 재료", "드레싱", "견과류 또는 토핑"],
        "dessert": ["주재료", "설탕", "버터 또는 크림", "계란 또는 우유", "토핑"],
        "egg": ["계란", "채소", "소금", "후추", "소스"],
        "snack": ["주재료", "소스", "기호에 따른 토핑"],
        "pancake": ["밀가루 또는 반죽", "계란", "물 또는 우유", "주재료", "기름"],
        "dumpling": ["만두피", "고기 또는 채소 속재료", "간장", "마늘", "대파"],
        "dip": ["주재료", "올리브유 또는 소스", "소금", "향신료"],
        "stir_fry": ["주재료", "채소", "간장 또는 소스", "마늘", "기름"],
    }

    ingredients = base_ingredients.get(
        category,
        ["주재료", "채소", "소금", "후추", "기호에 따른 소스"]
    )

    ingredient_text = "\n".join([f"- {item}" for item in ingredients])

    if category in ["stew", "soup"]:
        method = """
1. 재료를 깨끗하게 씻고 먹기 좋은 크기로 자릅니다.
2. 냄비를 중불로 예열합니다.
3. 다진 마늘과 대파를 먼저 넣어 향을 냅니다.
4. 주재료를 넣고 2~3분 정도 가볍게 익힙니다.
5. 물 또는 육수를 넣고 센 불에서 끓입니다.
6. 끓기 시작하면 중불로 줄이고 10분 정도 더 끓입니다.
7. 소금, 간장, 고춧가루 등으로 간을 맞춥니다.
8. 재료가 충분히 익으면 불을 끄고 그릇에 담습니다.
"""
    elif category in ["rice"]:
        method = """
1. 밥과 주재료를 준비합니다.
2. 채소나 고기는 먹기 좋은 크기로 손질합니다.
3. 팬을 중불로 예열한 뒤 기름을 약간 두릅니다.
4. 주재료와 채소를 볶습니다.
5. 밥을 넣고 소스와 함께 골고루 섞습니다.
6. 간을 확인한 뒤 부족하면 소금이나 간장으로 조절합니다.
7. 마지막에 참기름이나 고명을 올려 마무리합니다.
"""
    elif category in ["noodle", "pasta"]:
        method = """
1. 냄비에 물을 넉넉히 넣고 끓입니다.
2. 물이 끓으면 면을 넣고 적당히 삶습니다.
3. 팬에 기름을 두르고 마늘, 양파, 주재료를 볶습니다.
4. 소스 또는 육수를 넣고 중불에서 끓입니다.
5. 삶은 면을 넣고 소스와 골고루 섞습니다.
6. 간을 확인하고 필요하면 소금이나 후추를 추가합니다.
7. 고명이나 치즈, 허브를 올려 마무리합니다.
"""
    elif category in ["fried"]:
        method = """
1. 주재료를 먹기 좋은 크기로 자릅니다.
2. 소금과 후추로 밑간을 합니다.
3. 튀김가루 또는 빵가루를 골고루 묻힙니다.
4. 기름을 중불로 예열합니다.
5. 재료를 넣고 겉이 노릇해질 때까지 튀기거나 굽습니다.
6. 키친타월에 올려 기름을 뺍니다.
7. 소스와 함께 접시에 담아 완성합니다.
"""
    elif category in ["dessert", "bread", "pancake"]:
        method = """
1. 필요한 재료를 계량해 준비합니다.
2. 큰 볼에 반죽 재료를 넣고 골고루 섞습니다.
3. 팬이나 오븐을 예열합니다.
4. 반죽을 적당한 크기로 올립니다.
5. 겉면이 노릇해질 때까지 굽습니다.
6. 토핑이나 소스를 올려 마무리합니다.
"""
    else:
        method = """
1. 재료를 깨끗하게 씻고 먹기 좋은 크기로 손질합니다.
2. 팬이나 냄비를 중불로 예열합니다.
3. 기름을 약간 두르고 향이 나는 재료를 먼저 볶습니다.
4. 주재료를 넣고 충분히 익힙니다.
5. 소스나 양념을 넣고 골고루 섞습니다.
6. 간을 확인하고 부족하면 소금이나 후추를 추가합니다.
7. 그릇에 담고 토핑을 올려 완성합니다.
"""

    extra = ""

    if user_request:
        req = user_request.strip()

        if "맵" in req or "매운" in req:
            extra += """
🌶️ 요청 반영: 더 맵게 만들기
- 고춧가루나 매운 소스를 조금씩 추가하세요.
- 청양고추를 넣으면 깔끔한 매운맛이 납니다.
- 처음부터 많이 넣지 말고 마지막에 조금씩 조절하는 것이 좋습니다.
"""

        if "다이어트" in req or "칼로리" in req or "살" in req:
            extra += """
🥗 요청 반영: 다이어트 버전
- 기름 사용량을 줄이세요.
- 튀김보다 굽기, 삶기, 찌기 방식을 사용하세요.
- 소스와 소금 사용량을 줄이고 채소를 늘리면 좋습니다.
"""

        if "초보" in req or "쉽게" in req:
            extra += """
🔰 요청 반영: 초보자용 설명
- 재료 손질 → 예열 → 주재료 익히기 → 간 맞추기 순서만 기억하면 됩니다.
- 불은 처음부터 강하게 하지 말고 중불 위주로 조리하세요.
"""

        if "1인분" in req:
            extra += """
🍴 요청 반영: 1인분 기준
- 주재료는 한 줌 정도로 줄이세요.
- 물이나 육수는 약 300~400ml 정도부터 시작하세요.
- 양념은 1작은술 단위로 조금씩 넣어 간을 맞추세요.
"""

    if extra == "":
        extra = """
💡 추가 팁
- 음식 사진만으로 정확한 재료를 모두 알 수는 없으므로 결과는 추천 정보로 활용하세요.
- 실제 음식 스타일에 따라 양념과 재료를 조절하면 좋습니다.
"""

    return f"""
🍽️ 음식명: {food_name}

사진을 분석한 결과, 이 음식은 **{food_name}**일 가능성이 높습니다.

🛒 준비 재료
{ingredient_text}

👨‍🍳 초보자용 조리 방법
{method}

{extra}
"""


# =========================================================
# 10. 사용자 메시지 처리
# =========================================================
def process_message(image, user_text, chat_history, current_food_name):
    if chat_history is None:
        chat_history = []

    user_text = user_text.strip() if user_text else ""

    if image is None and user_text == "":
        return chat_history, current_food_name, None, ""

    if image is not None:
        top_foods = predict_food_with_clip(image, top_k=5)
        selected_food = top_foods[0]

        user_message = "음식 사진을 업로드했습니다."
        if user_text:
            user_message += f"\n\n{user_text}"

        assistant_message = generate_recipe(selected_food, user_text)

        chat_history.append({
            "role": "user",
            "content": user_message
        })

        chat_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return chat_history, selected_food["ko"], None, ""

    if image is None and user_text != "":
        if not current_food_name:
            assistant_message = "먼저 음식 사진을 업로드해주세요. 사진을 분석한 뒤 추가 질문에 답할 수 있습니다."

            chat_history.append({
                "role": "user",
                "content": user_text
            })

            chat_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            return chat_history, current_food_name, None, ""

        fake_food = {
            "ko": current_food_name,
            "en": current_food_name,
            "category": "stew",
            "score": 1.0,
        }

        assistant_message = generate_recipe(fake_food, user_text)

        chat_history.append({
            "role": "user",
            "content": user_text
        })

        chat_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return chat_history, current_food_name, None, ""

    return chat_history, current_food_name, None, ""


# =========================================================
# 11. UI
# =========================================================
with gr.Blocks() as demo:
    gr.Markdown("""
    # 🍳 AI 음식 레시피 챗봇

    음식 사진과 요청사항을 함께 입력하면  
    AI가 음식명을 추정하고 한국어 레시피를 대화형으로 제공합니다.
    """)

    current_food_state = gr.State("")

    chatbot = gr.Chatbot(
        label="대화",
        height=600,
    )

    with gr.Row():
        image_input = gr.Image(
            type="pil",
            label="사진 첨부",
        )

    with gr.Row():
        user_input = gr.Textbox(
            label="메시지 입력",
            placeholder="예: 이 음식이 뭐야? 레시피 알려줘 / 1인분 기준으로 알려줘 / 더 맵게 만들어줘",
            lines=2,
            scale=5,
        )

        send_button = gr.Button(
            "전송",
            scale=1,
        )

    send_button.click(
        fn=process_message,
        inputs=[
            image_input,
            user_input,
            chatbot,
            current_food_state,
        ],
        outputs=[
            chatbot,
            current_food_state,
            image_input,
            user_input,
        ],
    )

    user_input.submit(
        fn=process_message,
        inputs=[
            image_input,
            user_input,
            chatbot,
            current_food_state,
        ],
        outputs=[
            chatbot,
            current_food_state,
            image_input,
            user_input,
        ],
    )


demo.launch()