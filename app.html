"""
Kit de Reputación - v0.1
========================
Generador de respuestas a reseñas para restaurantes.

Cómo arrancarlo:
    1. Tener Python 3 instalado (ya lo tienes)
    2. Instalar dependencias:  pip install -r requirements.txt
    3. Copiar .env.example a .env y poner tu API key de DeepSeek
    4. Ejecutar:  streamlit run app.py

Se abre solo en http://localhost:8501
"""

import os
import html
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# 1. CONFIGURACIÓN INICIAL
# ============================================================

# Cargar variables desde el archivo .env (ahí va la API key, nunca en el código)
load_dotenv()

# Configurar la pestaña del navegador
st.set_page_config(
    page_title="Kit de Reputación",
    page_icon="⭐",
    layout="centered",
)

# Cliente de DeepSeek (usa la misma librería que OpenAI, solo cambia la URL)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
cliente = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
) if DEEPSEEK_API_KEY else None


# ============================================================
# 2. LOS TONOS DE VOZ DISPONIBLES
# ============================================================
# Cada tono tiene una descripción (para el selector) y una guía de estilo
# (para el prompt). Podéis añadir más tonos o ajustar estos.

TONOS = {
    "Cercano": {
        "descripcion": "Cálido y humano, como con un cliente habitual.",
        "guia": "Usa 'nosotros', tutea, menciona detalles personales si los hay.",
    },
    "Formal": {
        "descripcion": "Educado y distante, ideal para alta gama.",
        "guia": "Usa 'usted', evita coloquialismos, mantén distancia profesional.",
    },
    "Empático": {
        "descripcion": "Comprensivo con las quejas, ofrece solución.",
        "guia": "Reconoce el problema, ofrece solución concreta, deja contacto si procede.",
    },
    "Divertido": {
        "descripcion": "Cercano con humor, para ambiente juvenil.",
        "guia": "Puedes usar emojis con moderación, ironía amable, guiños al sector.",
    },
    "Profesional": {
        "descripcion": "Neutro y serio, apto para cualquier sitio.",
        "guia": "Tono corporativo, sin coloquialismos, directo al grano.",
    },
}


# ============================================================
# 3. EL PROMPT MAESTRO (el cerebro de la IA)
# ============================================================

def construir_prompt(reseña: str, tono: str, valoracion: int) -> str:
    """Construye el texto que se envía a DeepSeek.

    Cuantos más ejemplos (few-shot) pongáis aquí, mejor saldrán
    las respuestas. Esta es la parte que tenéis que iterar en las
    próximas semanas: con vuestras 30 reseñas + 30 respuestas modelo
    de la semana 2, alimentáis esta sección.
    """

    info_tono = TONOS[tono]

    # Ejemplos de respuestas de calidad (en español de España, hostelería)
    # TODO: Ampliar con más ejemplos reales en las próximas semanas.
    ejemplos = """
EJEMPLOS DE RESPUESTAS DE CALIDAD (español de España, hostelería):

Reseña positiva (5 estrellas): "Comida riquísima, el camarero Juan fue un sol. Volveremos seguro."
Respuesta modelo: "¡Muchas gracias por tu comentario! Nos alegra muchísimo que disfrutarais tanto de la comida y que Juan os hiciera sentir en casa. Ya sabes dónde encontrarnos cuando queráis repetir. ¡Os esperamos!"

Reseña neutra (3 estrellas): "El sitio está bien pero tardaron mucho en traer la cuenta."
Respuesta modelo: "Gracias por tu opinión. Lamentamos la espera con la cuenta, estamos ajustando el personal en horas pico para que no vuelva a ocurrir. Esperamos verte de nuevo para demostrarte que podemos hacerlo mejor."

Reseña negativa (1 estrella): "La paella estaba fría y el camarero fue borde."
Respuesta modelo: "Sentimos muchísimo que tu experiencia no fuera la que merecías. La paleta debió servirse a temperatura correcta y lamento el trato que recibiste: no es nuestro estándar. Si quieres escribirnos a info@tudominio.com nos gustaría compensarte. Gracias por avisarnos."
"""

    prompt = f"""Eres un experto en reputación online para restaurantes en España.
Tu tarea es redactar respuestas a reseñas de clientes en Google Maps, TripAdvisor y similares.

REGLAS:
1. Responde en español de España, no español neutro ni latinoamericano.
2. Adapta el tono exactamente a: {tono}. {info_tono['guia']}
3. La respuesta debe ser de 2-4 frases, nunca más de 80 palabras.
4. Si la reseña menciona a un empleado o plato específico, nómbralo en la respuesta.
5. Si la valoración es baja (1-2 estrellas), reconoce el problema y ofrece solución concreta.
6. Si la valoración es alta (4-5 estrellas), agradece y menciona algo específico del comentario.
7. No uses frases hechas tipo "Gracias por tu feedback" o "Lamentamos los inconvenientes". Suena a robot.
8. No inventes datos (nombres, platos, descuentos) que no estén en la reseña original.

VALORACIÓN DEL CLIENTE: {valoracion} de 5 estrellas.
TONO SELECCIONADO: {tono}

{ejemplos}

RESEÑA A LA QUE DEBES RESPONDER:
"{reseña}"

Genera 3 respuestas distintas, numeradas como [1], [2], [3], cada una con un enfoque
ligeramente diferente pero respetando el tono. Solo las respuestas, nada más."""

    return prompt


def generar_respuestas(reseña: str, tono: str, valoracion: int) -> list:
    """Llama a la API de DeepSeek y devuelve una lista con 3 respuestas."""

    if not cliente:
        raise RuntimeError("Falta la API key de DeepSeek en el archivo .env")

    prompt = construir_prompt(reseña, tono, valoracion)

    respuesta = cliente.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "Eres un redactor profesional especializado en reputación "
                           "online de hostelería en España.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,  # permitimos algo de creatividad para que salgan 3 versiones distintas
        max_tokens=800,
    )

    texto = respuesta.choices[0].message.content

    # Parsear las 3 respuestas separadas por [1], [2], [3]
    respuestas = []
    partes = texto.split("[1]")
    if len(partes) > 1:
        resto = partes[1]
        bloques = resto.split("[2]")
        respuestas.append(bloques[0].strip())
        if len(bloques) > 1:
            sub = bloques[1].split("[3]")
            respuestas.append(sub[0].strip())
            if len(sub) > 1:
                respuestas.append(sub[1].strip())

    # Si el modelo no respetó el formato, devolvemos el texto tal cual
    if not respuestas:
        respuestas = [texto.strip()]

    # Asegurar 3 respuestas (rellenar con aviso si faltan)
    while len(respuestas) < 3:
        respuestas.append("(no se generó respuesta)")

    return respuestas[:3]


# ============================================================
# 4. BOTÓN DE COPIAR AL PORTAPAPELES
# ============================================================

def boton_copiar(texto: str, key_id: str):
    """Muestra un botón HTML/JS que copia el texto al portapapeles.

    Funciona en localhost y en HTTPS. En HTTP plano podría fallar
    por restricciones del navegador, pero vuestro VPS llevará HTTPS.
    """
    texto_escapado = html.escape(texto)
    components.html(
        f"""
        <button
          onclick="copy_{key_id}(this)"
          style="
            background-color: #FF4B4B;
            color: white;
            border: none;
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            margin-bottom: 10px;
          "
        >
          📋 Copiar al portapapeles
        </button>
        <textarea id="hidden_{key_id}" style="display:none">{texto_escapado}</textarea>
        <script>
          function copy_{key_id}(btn) {{
            const text = document.getElementById('hidden_{key_id}').value;
            if (navigator.clipboard && navigator.clipboard.writeText) {{
              navigator.clipboard.writeText(text).then(() => {{
                const original = btn.innerText;
                btn.innerText = '✓ Copiado';
                setTimeout(() => {{ btn.innerText = original; }}, 2000);
              }}).catch(err => {{
                fallbackCopy(text, btn);
              }});
            }} else {{
              fallbackCopy(text, btn);
            }}
          }}
          function fallbackCopy(text, btn) {{
            const ta = document.createElement('textarea');
            ta.value = text;
            document.body.appendChild(ta);
            ta.select();
            try {{
              document.execCommand('copy');
              const original = btn.innerText;
              btn.innerText = '✓ Copiado';
              setTimeout(() => {{ btn.innerText = original; }}, 2000);
            }} catch(e) {{
              alert('Copia manual: ' + text);
            }}
            document.body.removeChild(ta);
          }}
        </script>
        """,
        height=60,
    )


# ============================================================
# 5. LA INTERFAZ DE USUARIO (lo que ve el restaurante)
# ============================================================

st.title("⭐ Kit de Reputación")
st.markdown("Generador de respuestas a reseñas para restaurantes")
st.divider()

# Aviso si falta la API key
if not DEEPSEEK_API_KEY:
    st.error(
        "⚠️ No se encontró la API key de DeepSeek.\n\n"
        "Crea un archivo `.env` en la misma carpeta que `app.py` con esta línea:\n\n"
        "`DEEPSEEK_API_KEY=tu_clave_aqui`"
    )
    st.stop()

# --- Entradas del usuario ---
st.subheader("Datos de la reseña")

valoracion = st.slider(
    "⭐ Valoración del cliente (estrellas)",
    min_value=1, max_value=5, value=5, step=1,
)

reseña = st.text_area(
    "📝 Pega aquí la reseña que recibiste",
    height=120,
    placeholder="Ej: La comida estaba deliciosa pero tardaron bastante en atender...",
)

tono = st.selectbox(
    "🎭 Tono de la respuesta",
    list(TONOS.keys()),
    format_func=lambda t: f"{t} — {TONOS[t]['descripcion']}",
)

# --- Botón principal ---
if st.button("🚀 Generar 3 respuestas", type="primary", use_container_width=True):

    if not reseña.strip():
        st.warning("Pega primero la reseña del cliente.")
    else:
        with st.spinner("Generando respuestas..."):
            try:
                respuestas = generar_respuestas(reseña, tono, valoracion)

                st.divider()
                st.subheader("✍️ Respuestas sugeridas (editables)")
                st.caption("Puedes editar el texto antes de copiarlo. El botón copia la versión editada.")

                for i, resp in enumerate(respuestas, 1):
                    st.markdown(f"**Opción {i}**")
                    editada = st.text_area(
                        label=f"Respuesta {i}",
                        value=resp,
                        height=140,
                        key=f"resp_{i}",
                        label_visibility="collapsed",
                    )
                    boton_copiar(editada, f"btn_{i}")
                    st.markdown("---")

            except Exception as e:
                st.error(f"❌ Error al generar las respuestas: {e}")
                st.info(
                    "Revisa que:\n"
                    "- La API key de DeepSeek es válida.\n"
                    "- Tienes crédito en tu cuenta de DeepSeek.\n"
                    "- Tienes conexión a internet."
                )


# --- Pie de página ---
st.markdown("---")
st.caption("Kit de Reputación v0.1 — Prototipo en desarrollo")
