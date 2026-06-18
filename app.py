import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from deep_translator import GoogleTranslator

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Recipe Predictor",
    page_icon="🍽️",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background-image: url('https://images.unsplash.com/photo-1556910103-1c02745aae4d?w=1920&q=80');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
}

.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(135deg, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.4) 100%);
    z-index: 0;
}

.stApp > * {
    position: relative;
    z-index: 1;
}

.header-box {
    background: linear-gradient(135deg, rgba(255,75,75,0.9) 0%, rgba(255,123,84,0.9) 100%);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 15px 40px rgba(255,75,75,0.3);
}

.big-title {
    font-size: 48px;
    font-weight: 800;
    color: white;
    margin: 0;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
}

.subtitle {
    color: rgba(255,255,255,0.9);
    font-size: 16px;
    margin-top: 8px;
}

.stTextArea textarea {
    border-radius: 15px;
    border: 2px solid #e0e0e0;
    padding: 18px;
    font-size: 16px;
    background: #fafafa;
    transition: all 0.3s ease;
}

.stTextArea textarea:focus {
    border-color: #ff4b4b;
    box-shadow: 0 0 20px rgba(255,75,75,0.15);
}

.stButton>button {
    background: linear-gradient(135deg, #ff4b4b 0%, #ff7b54 100%);
    color: white;
    border-radius: 50px;
    height: 55px;
    font-weight: 700;
    font-size: 18px;
    width: 100%;
    border: none;
    letter-spacing: 1px;
    transition: all 0.3s ease;
    box-shadow: 0 8px 25px rgba(255,75,75,0.4);
}

.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(255,75,75,0.6);
}

.cuisine-badge {
    display: inline-block;
    background: linear-gradient(135deg, #ff4b4b 0%, #ff7b54 100%);
    color: white;
    padding: 12px 30px;
    border-radius: 50px;
    font-size: 22px;
    font-weight: 700;
    box-shadow: 0 8px 25px rgba(255,75,75,0.4);
}

.prediction-card {
    background: white;
    padding: 18px;
    border-radius: 15px;
    margin: 8px 0;
    border-left: 4px solid #ff4b4b;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

.recipe-name-box {
    background: linear-gradient(135deg, #fff5f5 0%, #fff0ed 100%);
    padding: 20px;
    border-radius: 15px;
    margin: 12px 0;
    border-left: 5px solid #ff4b4b;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

.recipe-desc-box {
    background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
    padding: 20px;
    border-radius: 15px;
    margin: 12px 0;
    border-left: 5px solid #667eea;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

.recipe-ingredients-box {
    background: linear-gradient(135deg, #fff9f0 0%, #fff5e6 100%);
    padding: 20px;
    border-radius: 15px;
    margin: 12px 0;
    border-left: 5px solid #ffa726;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

.recipe-instructions-box {
    background: linear-gradient(135deg, #f0fff4 0%, #e6ffe6 100%);
    padding: 20px;
    border-radius: 15px;
    margin: 12px 0;
    border-left: 5px solid #4caf50;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

.recipe-time-box {
    background: linear-gradient(135deg, #fff5f5 0%, #ffe6e6 100%);
    padding: 20px;
    border-radius: 15px;
    margin: 12px 0;
    border-left: 5px solid #e91e63;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    text-align: center;
}

.recipe-image-box {
    background: linear-gradient(135deg, #fff8e1 0%, #fff3e0 100%);
    padding: 20px;
    border-radius: 15px;
    margin: 12px 0;
    border-left: 5px solid #ff9800;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    text-align: center;
}

.section-icon {
    font-size: 28px;
    margin-right: 8px;
}

.section-title {
    font-size: 18px;
    font-weight: 700;
    color: #333;
    display: flex;
    align-items: center;
}

.ingredient-item {
    background: white;
    padding: 8px 15px;
    border-radius: 25px;
    display: inline-block;
    margin: 4px 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    font-size: 14px;
}

.stImage img {
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.footer {
    text-align: center;
    padding: 20px;
    color: rgba(255,255,255,0.7);
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

model = Sequential([
    Dense(128, activation='relu', input_shape=(500,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(56, activation='softmax')
])

model.load_weights("models/recipe_model.weights.h5")

# =====================================================
# LOAD FILES
# =====================================================

with open("models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

df = pd.read_csv("models/recipes.csv")

# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# =====================================================
# TRANSLATE
# =====================================================

def translate(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(str(text))
    except:
        return text

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="header-box">
    <h1 class="big-title">🍽️ AI Recipe Predictor</h1>
    <p class="subtitle">✨ Describe any dish — Discover its Cuisine & Get the Perfect Recipe</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# INPUT
# =====================================================

st.markdown("""
<div style="background: rgba(255,255,255,0.95); padding: 30px; border-radius: 20px; 
            box-shadow: 0 15px 40px rgba(0,0,0,0.2); margin-bottom: 25px;">
    <div style="text-align:center; margin-bottom:15px;">
        <h3 style="color:#333; font-weight:600;">📝 What are you cooking today?</h3>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    user_input = st.text_area(
        "Recipe Description",
        placeholder="e.g., Tender mutton pieces slow cooked with aromatic spices, basmati rice, saffron, and caramelized onions...",
        height=140,
        label_visibility="collapsed"
    )
    predict = st.button("🔮 Predict Cuisine & Get Recipe")

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PREDICTION
# =====================================================

if predict:

    if user_input.strip() == "":
        st.warning("⚠️ Please enter a recipe description first!")
    else:

        with st.spinner("🍳 Analyzing your recipe..."):
            cleaned = clean_text(user_input)
            X_input = tfidf.transform([cleaned]).toarray()
            prediction = model.predict(X_input, verbose=0)
            index = np.argmax(prediction)
            cuisine = encoder.inverse_transform([index])[0]
            confidence = float(np.max(prediction) * 100)

        # =============================================
        # RESULT BOX
        # =============================================

        st.markdown("""
        <div style="background: rgba(255,255,255,0.95); padding: 30px; border-radius: 20px; 
                    box-shadow: 0 15px 40px rgba(0,0,0,0.2); margin-bottom: 25px;">
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<h3 style="color: #FF6B6B;">🎯 Predicted Cuisine</h3>', unsafe_allow_html=True)

            cuisine_colors = {
                'italian': '#FF6B6B',
                'mexican': '#FFA500',
                'chinese': '#FFD700',
                'indian': '#FF8C42',
                'japanese': '#FF69B4',
                'thai': '#32CD32',
                'french': '#9370DB',
                'american': '#4169E1',
            }

            badge_color = cuisine_colors.get(cuisine.lower(), '#00CED1')

            st.markdown(
                f'<div class="cuisine-badge" style="background-color: {badge_color}; color: white; padding: 12px 20px; border-radius: 10px; font-size: 20px; font-weight: bold; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">'
                f'{cuisine}'
                f'</div>',
                unsafe_allow_html=True
            )

        with col_b:
            st.markdown('<h3 style="color: #FF6B6B;">📊 Confidence Level</h3>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="padding:10px;">
                <h2 style="color:#ff4b4b; font-size:40px; font-weight:800; margin:0;">{confidence:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            st.progress(min(confidence / 100, 1.0))

        # Top 3 Predictions
        st.markdown('<h3 style="color: #FF6B6B;">📈 Top 3 Predictions</h3>', unsafe_allow_html=True)
        top3_idx = np.argsort(prediction[0])[-3:][::-1]

        cols = st.columns(3)
        for i, (rank, idx) in enumerate(zip(range(3), top3_idx)):
            cuisine_name = encoder.inverse_transform([idx])[0]
            conf = float(prediction[0][idx] * 100)

            if rank == 0:
                emoji, color = "🥇", "#ff4b4b"
            elif rank == 1:
                emoji, color = "🥈", "#ffa726"
            else:
                emoji, color = "🥉", "#667eea"

            with cols[i]:
                st.markdown(f"""
                <div class="prediction-card" style="border-left-color:{color};">
                    <h4 style="margin:0; color:{color};">{emoji} {cuisine_name}</h4>
                    <p style="margin:5px 0 0 0; font-size:18px; font-weight:600;">{conf:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # =============================================
        # FIND BEST MATCHING RECIPE
        # =============================================

        recipes = df[df['cuisine'] == cuisine]

        if len(recipes) > 0:

            best_recipe_idx = None
            best_recipe_score = 0

            for idx, row in recipes.iterrows():
                recipe_text = str(row['name']) + " " + str(row['description']) + " " + str(row['ingredients'])
                recipe_text = clean_text(recipe_text)
                recipe_vector = tfidf.transform([recipe_text]).toarray()
                similarity = np.dot(recipe_vector, X_input.T)[0][0]

                if similarity > best_recipe_score:
                    best_recipe_score = similarity
                    best_recipe_idx = idx

            if best_recipe_idx is not None and best_recipe_score > 0:
                recipe = recipes.loc[best_recipe_idx]
            else:
                recipe = recipes.sample(1).iloc[0]

            # =============================================
            # RECIPE IMAGE
            # =============================================

            recipe_name = str(recipe['name']).lower()

            image_folder = "images/data"
            found_image = None

            if os.path.exists(image_folder):

                recipe_words = set(re.findall(r'\w+', recipe_name))

                best_score = 0

                for file in os.listdir(image_folder):

                    file_path = os.path.join(image_folder, file)

                    if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):

                        file_name = os.path.splitext(file)[0].lower()

                        # replace _ and -
                        file_name = file_name.replace("_", " ").replace("-", " ")

                        file_words = set(re.findall(r'\w+', file_name))

                        score = len(recipe_words.intersection(file_words))

                        if score > best_score:
                            best_score = score
                            found_image = file_path

            # DISPLAY IMAGE
            st.markdown('<div class="recipe-image-box">', unsafe_allow_html=True)

            st.markdown("""
            <div class="section-title" style="justify-content:center;">
                <span class="section-icon">🖼️</span> Recipe Image
            </div>
            """, unsafe_allow_html=True)

            col_img1, col_img2, col_img3 = st.columns([1, 2, 1])

            with col_img2:

                if found_image:

                    try:
                        img = Image.open(found_image)

                        img = img.resize((500, 500))

                        st.image(img)

                        st.markdown(
                            f"<p style='text-align:center;'>🍽️ {recipe['name']}</p>",
                            unsafe_allow_html=True
                        )

                    except Exception as e:
                        st.error(f"Image error: {e}")

                else:
                    st.warning("No matching image found.")

            st.markdown("</div>", unsafe_allow_html=True)

            # RECIPE NAME
            st.markdown(f"""
            <div class="recipe-name-box">
                <div class="section-title">
                    <span class="section-icon">🍴</span> Recipe Name
                </div>
                <h2 style="color:#ff4b4b; margin:10px 0 0 0; font-size:24px;">{translate(recipe['name'])}</h2>
            </div>
            """, unsafe_allow_html=True)

            # DESCRIPTION
            st.markdown(f"""
            <div class="recipe-desc-box">
                <div class="section-title">
                    <span class="section-icon">📝</span> Description
                </div>
                <p style="color:#444; line-height:1.8; margin-top:10px; font-size:15px;">{translate(recipe['description'])}</p>
            </div>
            """, unsafe_allow_html=True)

            # INGREDIENTS
            ingredients_text = translate(recipe['ingredients'])
            ingredients_text = re.sub(r'\s+', ' ', ingredients_text)
            ingredients_list = [i.strip() for i in ingredients_text.split(',') if i.strip()]

            ingredients_html = """
            <div class="recipe-ingredients-box">
                <div class="section-title">
                    <span class="section-icon">🥘</span> Ingredients
                </div>
                <div style="margin-top:12px;">
            """
            for item in ingredients_list:
                ingredients_html += f'<span class="ingredient-item">🥄 {item}</span>'
            ingredients_html += '</div></div>'

            st.markdown(ingredients_html, unsafe_allow_html=True)

            # INSTRUCTIONS
            if 'instructions' in df.columns:
                instructions = translate(recipe['instructions'])
                instructions = re.sub(r'\s+', ' ', instructions)
                steps = [s.strip() for s in instructions.split('.') if s.strip()]

                instructions_html = """
                <div class="recipe-instructions-box">
                    <div class="section-title">
                        <span class="section-icon">👨‍🍳</span> Step by Step Instructions
                    </div>
                    <ol style="margin-top:12px; line-height:2;">
                """
                for step in steps[:10]:
                    if step:
                        instructions_html += f'<li style="color:#333; font-size:15px;">{step}.</li>'
                instructions_html += '</ol></div>'

                st.markdown(instructions_html, unsafe_allow_html=True)

            # PREP TIME
            if 'prep_time' in df.columns:
                st.markdown(f"""
                <div class="recipe-time-box">
                    <div class="section-title" style="justify-content:center;">
                        <span class="section-icon">⏰</span> Preparation Time
                    </div>
                    <h3 style="color:#e91e63; margin:10px 0 0 0; font-size:28px;">{recipe['prep_time']} minutes</h3>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.warning(f"😔 No recipes found for {cuisine} cuisine.")

# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    <hr style="border-color: rgba(255,255,255,0.2);">
    <p>🍕 AI Recipe Predictor | Built with ❤️ using Streamlit + TensorFlow + Machine Learning</p>
    <p>Supports 56+ cuisines from around the world 🌍</p>
</div>
""", unsafe_allow_html=True)