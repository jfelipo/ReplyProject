"""
Kit de Reputación - v0.2
========================
Generador de respuestas a reseñas para restaurantes.

Funciona con Groq (gratis) o DeepSeek (pago, mejor calidad).

Cómo arrancarlo:
    1. Tener Python 3 instalado
    2. pip install -r requirements.txt
    3. Configurar la API key en .env (local) o en Secrets (Streamlit Cloud)
    4. streamlit run app.py
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

# Cargar variables desde .env (local). No hace nada si no hay .env.
load_dotenv()

# Obtener la API key:
# - En Streamlit Cloud: lee de st.secrets (configurado en el panel web)
# - En local: lee del archivo .env
try:
    API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except:
    API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Configurar la página
st.set_page_config(
    page_title="Kit de Reputación",
    page_icon="*",
    layout="centered",
)

# Cliente de Groq (usa la misma librería que OpenAI, solo cambia la URL)
cliente = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1",
) if API_KEY else None


# ============================================================
# 2. LOS TONOS DE VOZ DISPONIBLES
# ============================================================

TONOS = {
    "Cercano": {
        "descripción": "Cálido y humano, como con un cliente habitual.",
        "guía": "Usa 'nosotros', tutea, menciona detalles personales si los hay. Tono humano, no corporativo.",
    },
    "Formal": {
        "descripción": "Educado y distante, ideal para alta gama.",
        "guía": "Usa 'usted', lenguaje cuidado, mantén distancia profesional.",
    },
    "Empático": {
        "descripción": "Comprensivo con las quejas, conecta emocionalmente.",
        "guía": "Reconoce la emoción del cliente (frustración, decepción), usa tono humano y cálido, ofrece solución concreta y proactiva (no 'contacte con nosotros').",
    },
    "Divertido": {
        "descripción": "Cercano con humor, para ambiente juvenil.",
        "guía": "Puedes usar emojis con moderación, ironía amable, guiños al sector.",
    },
    "Profesional": {
        "descripción": "Neutro y serio, apto para cualquier sitio.",
        "guía": "Tono corporativo pero natural, sin coloquialismos, directo al grano.",
    },
}


# ============================================================
# 3. EL PROMPT MAESTRO
# ============================================================

def construir_prompt(reseña: str, tono: str, valoración: int) -> str:
    """Construye el texto que se envía a la IA."""

    info_tono = TONOS[tono]

    ejemplos = """
EJEMPLOS DE RESPUESTAS DE CALIDAD (español de España, hostelería):

Reseña positiva (5 estrellas): "Comida riquísima, el camarero Juan fue un sol. Volveremos seguro."
Respuesta modelo: "¡Muchas gracias por tu comentario! Nos alegra muchísimo que disfrutarais tanto de la comida y que Juan os hiciera sentir en casa. Ya sabes dónde encontrarnos cuando queráis repetir. ¡Os esperamos!"

Reseña neutra (3 estrellas): "El sitio está bien pero tardaron mucho en traer la cuenta."
Respuesta modelo: "Gracias por tu opinión. Lamentamos la espera con la cuenta, estamos ajustando el personal en horas pico para que no vuelva a ocurrir. Esperamos verte de nuevo para demostrarte que podemos hacerlo mejor."

Reseña negativa (1 estrella): "La paella estaba fría y el camarero fue borde."
Respuesta modelo: "Sentimos muchísimo que tu experiencia no fuera la que merecías. La paella debió servirse a temperatura correcta y lamento el trato que recibiste: no es nuestro estándar. Nos gustaría compensarte: escríbenos a info@tudominio.com y te invitamos a volver para que veas cómo hacemos las cosas bien. Gracias por avisarnos."

Reseña negativa grave (1 estrella): "Teníamos reservado para celebrar nuestro aniversario. Al llegar nos pusieron en una mesa junto al servicio. El camarero nos trató con malos modos toda la cena. Arruinó nuestra celebración."
Respuesta modelo: "Nos duele enormemente leer esto. Teníais una reserva para una ocasión especial y no la respetamos: ni la mesa asignada ni el trato del personal estuvieron a la altura. No tiene excusa. Sabemos que una celebración como esa no se recupera, pero nos gustaría invitaros a una cena completa a nuestro cargo para que podáis celebrar como merecíais. Escribenos a info@tudominio.com. Lo sentimos de verdad."
"""

    prompt = f"""Eres un experto en reputación online para restaurantes en España.
Tu tarea es redactar respuestas a reseñas de clientes en Google Maps, TripAdvisor y similares.

REGLAS:
1. Responde en español de España, no español neutro ni latinoamericano.
2. Adapta el tono exactamente a: {tono}. {info_tono['guía']}
3. La respuesta debe ser de 2-4 frases, nunca más de 80 palabras.
4. Si la reseña menciona un detalle específico (nombre del camarero, plato, hora, reserva), nómbralo en la respuesta.
5. Si la valoración es baja (1-2 estrellas), reconoce el problema específico y ofrece una compensación concreta y proactiva (no "contacte con nosotros").
6. Si la valoración es alta (4-5 estrellas), agradece y menciona algo específico del comentario del cliente.
7. No uses frases hechas tipo "Gracias por tu feedback" o "Lamentamos los inconvenientes". Suena a robot.
8. No inventes datos (nombres, platos, descuentos) que no estén en la reseña original.
9. Suena humano, no a comunicado de empresa. El dueño del bar escribiría la respuesta, no el departamento de PR.

VALORACIÓN DEL CLIENTE: {valoración} de 5 estrellas.
TONO SELECCIONADO: {tono}

{ejemplos}

RESEÑA A LA QUE DEBES RESPONDER:
"{reseña}"

Genera 3 respuestas numeradas como [1], [2], [3], cada una con un ENFOQUE DISTINTO:
[1] Formal y profesional: tono serio, lenguaje cuidado.
[2] Cálida y empática: conecta emocionalmente con lo que el cliente sintió.
[3] Orientada a solución: ofrece una compensación concreta y proactiva.

Las tres deben respetar el tono general seleccionado, pero con esos enfoques diferenciados.
Solo las respuestas, nada más."""

    return prompt


def generar_respuestas(reseña: str, tono: str, valoración: int) -> list:
    """Llama a la API de Groq y devuelve una lista con 3 respuestas."""

    if not cliente:
        raise RuntimeError("Falta la API key. Configúrala en .env o en Streamlit Cloud Secrets.")

    prompt = construir_prompt(reseña, tono, valoración)

    respuesta = cliente.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "system",
                "content": "Eres un redactor profesional especializado en reputación "
                           "online de hostelería en España.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
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

    if not respuestas:
        respuestas = [texto.strip()]

    while len(respuestas) < 3:
        respuestas.append("(no se generó respuesta)")

    return respuestas[:3]


# ============================================================
# 4. BOTÓN DE COPIAR AL PORTAPAPELES
# ============================================================

def botón_copiar(texto: str, key_id: str):
    """Muestra un botón HTML/JS que copia el texto al portapapeles."""
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
          Copiar al portapapeles
        </button>
        <textarea id="hidden_{key_id}" style="display:none">{texto_escapado}</textarea>
        <script>
          function copy_{key_id}(btn) {{
            const text = document.getElementById('hidden_{key_id}').value;
            if (navigator.clipboard && navigator.clipboard.writeText) {{
              navigator.clipboard.writeText(text).then(() => {{
                const original = btn.innerText;
                btn.innerText = 'Copiado';
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
              btn.innerText = 'Copiado';
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
# 5. LA INTERFAZ DE USUARIO
# ============================================================

st.title("Kit de Reputación")
st.markdown("Generador de respuestas a reseñas para restaurantes")
st.divider()

if not API_KEY:
    st.error(
        "No se encontró la API key.\n\n"
        "Si estás en local: crea un archivo `.env` con:\n"
        "`DEEPSEEK_API_KEY=tu_clave_aqui`\n\n"
        "Si estás en Streamlit Cloud: ve a Settings → Secrets y añade:\n"
        "`DEEPSEEK_API_KEY = \"tu_clave_aqui\"`"
    )
    st.stop()

st.subheader("Datos de la reseña")

valoración = st.slider(
    "Valoración del cliente (estrellas)",
    min_value=1, max_value=5, value=5, step=1,
)

reseña = st.text_area(
    "Pega aquí la reseña que recibiste",
    height=120,
    placeholder="Ej: La comida estaba deliciosa pero tardaron bastante en atender...",
)

tono = st.selectbox(
    "Tono de la respuesta",
    list(TONOS.keys()),
    format_func=lambda t: f"{t} - {TONOS[t]['descripción']}",
)

if st.button("Generar 3 respuestas", type="primary", use_container_width=True):

    if not reseña.strip():
        st.warning("Pega primero la reseña del cliente.")
    else:
        with st.spinner("Generando respuestas..."):
            try:
                respuestas = generar_respuestas(reseña, tono, valoración)

                st.divider()
                st.subheader("Respuestas sugeridas (editables)")
                st.caption("Opción 1: formal. Opción 2: cálida. Opción 3: solución concreta. "
                           "Puedes editar antes de copiar.")

                for i, resp in enumerate(respuestas, 1):
                    st.markdown(f"**Opción {i}**")
                    editada = st.text_area(
                        label=f"Respuesta {i}",
                        value=resp,
                        height=140,
                        key=f"resp_{i}",
                        label_visibility="collapsed",
                    )
                    botón_copiar(editada, f"btn_{i}")
                    st.markdown("---")

            except Exception as e:
                st.error(f"Error al generar las respuestas: {e}")
                st.info(
                    "Revisa que:\n"
                    "- La API key es válida.\n"
                    "- Tienes crédito en tu cuenta.\n"
                    "- Tienes conexión a internet."
                )


st.markdown("---")
st.caption("Kit de Reputación v0.2 - Prototipo en desarrollo")
