import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def display_df(uploaded_file):

    try:
        n_filas = 50  # Número de registros
        tipos_vehiculos = ['Automóvil', 'Camión', 'Motocicleta', 'Bus', 'Bicicleta']
        direcciones = ['Norte', 'Sur', 'Este', 'Oeste']
        estados_climaticos = ['Despejado', 'Lluvioso', 'Niebla', 'Nublado']

        data = {
            "id_vehiculo": [f"V-{i}" for i in range(1, n_filas + 1)],
            "tipo_vehiculo": [random.choice(tipos_vehiculos) for _ in range(n_filas)],
            "velocidad_kmh": [round(random.uniform(20, 120), 2) for _ in range(n_filas)],
            "hora_paso": [(datetime.now() - timedelta(seconds=random.randint(0, 3600))).strftime('%Y-%m-%d %H:%M:%S') for _ in range(n_filas)],
            "direccion": [random.choice(direcciones) for _ in range(n_filas)],
            "carril": [random.randint(1, 4) for _ in range(n_filas)],
            "estado_climatico": [random.choice(estados_climaticos) for _ in range(n_filas)],
            "peso_vehiculo_kg": [random.randint(500, 30000) for _ in range(n_filas)],
            "matricula": [f"XXX-{random.randint(1000, 9999)}" for _ in range(n_filas)],
            "tiempo_en_carretera_min": [random.randint(1, 180) for _ in range(n_filas)],
        }

        # Crear el DataFrame
        df = pd.DataFrame(data)

        # Mostrar el DataFrame en Streamlit
        st.title("Registros de Vehículos en la Carretera")
        st.write("Este es un reporte de los vehículos que pasan por una carretera.")

        # Usar st.dataframe para mostrarlo de forma interactiva
        st.dataframe(df)

        # Alternativamente, usar st.table para una visualización más estática
        st.write("Vista estática:")
        st.table(df.head(10))  # Mostrar las primeras 10 filas
    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
        print(e)

def imagen(uploaded_file):
    from IPython import display
    display.clear_output()
    import ultralytics
    from IPython import display
    display.clear_output()
    import supervision as sv
    from ultralytics import YOLO
    model = YOLO("yolov8x.pt")
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        SOURCE_VIDEO_PATH = temp_video.name  # Ruta del archivo temporal
    #SOURCE_VIDEO_PATH=uploaded_file
    
    # dict maping class_id to class_name
    CLASS_NAMES_DICT = model.model.names

    # the class names we have chosen
    SELECTED_CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']

    # class ids matching the class names we have chosen
    SELECTED_CLASS_IDS = [
        {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
        for class_name
        in SELECTED_CLASS_NAMES
    ]
    
    # create frame generator
    generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    # create instance of BoxAnnotator and LabelAnnotator
    box_annotator = sv.BoxAnnotator(thickness=4)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.5, text_color=sv.Color.BLACK)
    # acquire first video frame
    
    iterator = iter(generator)
    frame = next(iterator)
    # model prediction on single frame and conversion to supervision Detections
    results = model(frame, verbose=False)[0]
    
    # convert to Detections
    detections = sv.Detections.from_ultralytics(results)
    # only consider class id from selected_classes define above
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]

    # format custom labels
    labels = [
        f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for confidence, class_id in zip(detections.confidence, detections.class_id)
    ]
    
    # annotate and display frame
    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)

    #%matplotlib inline
    #sv.plot_image(annotated_frame, (16, 16))
    st.image(annotated_frame, caption="Resultados de detección", use_container_width=True)


st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Bienvenidos a Transito! 👋")

st.sidebar.success("Esta app permite contar la cantidad de vehiculos.")
st.sidebar.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)

st.header("Cargar Video")
uploaded_file = st.file_uploader("Por favor sube tu video:", type= "mp4")


if uploaded_file is not None:
    try: 
        with st.spinner("Cargando la imagen..."):       
            display_df(uploaded_file)
            st.write("Input Processed")
            imagen(uploaded_file)
    # Código potencialmente problemático
    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
        print(e)


# Generar datos ficticios



