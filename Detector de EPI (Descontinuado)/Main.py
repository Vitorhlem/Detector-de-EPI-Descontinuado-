# Importing necessary libraries
from ultralytics import YOLO
import cv2
import time 
import os
import numpy as np
import supervision as sv # Adicionado para tracking e outras utilidades
from collections import defaultdict 

# --- CONFIGURAÇÃO DOS MODELOS ---
MODEL_PATH_1 = r'C:\Users\Vitor Lemes\Downloads\Detector_EPI_YOLO Melhorado - Copia\Detector_EPI_YOLO Melhorado - Copia\modelos_yolo\yolov8s_custom.pt'
MODEL_PATH_2 = r'C:\Users\Vitor Lemes\Downloads\Detector_EPI_YOLO Melhorado - Copia\Detector_EPI_YOLO Melhorado - Copia\modelos_yolo\Best (3).pt' 

# --- Carregar Modelo 1 ---
if not os.path.exists(MODEL_PATH_1):
    print(f"ERRO: Modelo 1 '{MODEL_PATH_1}' não encontrado. Verifique o caminho.")
    exit()
model1 = YOLO(MODEL_PATH_1)
print(f"Modelo 1 ('{os.path.basename(MODEL_PATH_1)}') carregado com sucesso.")
if hasattr(model1, 'names'):
    print(f"  Classes do Modelo 1: {model1.names}")
else:
    print("AVISO: Atributo 'names' não encontrado para Modelo 1. Verifique o modelo.")
    model1.names = {} 

# --- Carregar Modelo 2 ---
if not os.path.exists(MODEL_PATH_2):
    print(f"ERRO: Modelo 2 '{MODEL_PATH_2}' não encontrado. Verifique o caminho.")
    model2 = None 
else:
    model2 = YOLO(MODEL_PATH_2)
    print(f"Modelo 2 ('{os.path.basename(MODEL_PATH_2)}') carregado com sucesso.")
    if hasattr(model2, 'names'):
        print(f"  Classes do Modelo 2: {model2.names}")
    else:
        print("AVISO: Atributo 'names' não encontrado para Modelo 2. Verifique o modelo.")
        model2.names = {} 

# --- CONFIGURAÇÃO DA FONTE DE DETECÇÃO ---
VIDEO_SOURCE = 0 
DESIRED_WIDTH = 1920
DESIRED_HEIGHT = 1080

cap = cv2.VideoCapture(VIDEO_SOURCE)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)

if not cap.isOpened():
    print(f"ERRO: Não foi possível abrir a câmera com índice {VIDEO_SOURCE}.")
    exit()

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Câmera com índice {VIDEO_SOURCE} aberta.")
print(f"Resolução solicitada: {DESIRED_WIDTH}x{DESIRED_HEIGHT}")
print(f"Resolução atual da câmera: {actual_width}x{actual_height}")
if actual_width != DESIRED_WIDTH or actual_height != DESIRED_HEIGHT:
    print("AVISO: A câmera não suporta ou não aplicou a resolução solicitada. Usando resolução disponível.")

WINDOW_NAME = 'Detector de EPI com Tracking - Pressione ESC para sair' 
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

last_check_time = time.time()

# --- CONFIGURAÇÕES DE TRACKING E ASSOCIAÇÃO ---
tracker = sv.ByteTrack(frame_rate=30) 
box_annotator = sv.BoxAnnotator(thickness=2) 
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER, text_scale=0.5, text_thickness=1)

PERSON_CLASSES = ['person', 'worker'] 
IOU_THRESHOLD_ASSOCIATION = 0.1 

SAFETY_EPIS_OBRIGATORIOS = [
    'glasses', 'goggles', 
    'gloves', 'glove',     
    'helmet',              
    'safety-vest', 'vest',
    'mask',                
    'safety-boot', 'boots',
    'ear protectors', 'ear-protector' 
]

TRADUCOES_PARA_PORTUGUES = {
    'helmet': 'Capacete', 'hardhat': 'Capacete',
    'mask': 'Mascara', 'mascara': 'Mascara', 'welder mask': 'Mascara de Solda',
    'goggles': 'Oculos de Protecao', 'oculos': 'Oculos de Protecao',
    'glass': 'Oculos', 'glasses': 'Oculos', 
    'ear-protector': 'Protetor Auricular', 'protetor auricular': 'Protetor Auricular',
    'ear protectors': 'Protetor Auricular', 'earmuffs': 'Abafador',
    'vest': 'Colete', 'colete': 'Colete',
    'safety-vest': 'Colete de Seguranca',
    'gloves': 'Luvas', 'luvas': 'Luvas', 'glove': 'Luvas',
    'boots': 'Botas', 'botas': 'Botas', 'safety_shoes': 'Calcado de Seguranca', 'shoes': 'Calcado',
    'safety-boot': 'Bota de Seguranca',
    'person': 'Pessoa', 'pessoa': 'Pessoa',
    'worker': 'Trabalhador',
    'no-helmet': 'Sem Capacete', 'no helmet': 'Sem Capacete', 'no_helmet': 'Sem Capacete',
    'without helmet': 'Sem Capacete',
    'no-vest': 'Sem Colete', 'no vest': 'Sem Colete', 'no_vest': 'Sem Colete',
    'without vest': 'Sem Colete',
    'protective_suit': 'Traje de Protecao',
    'chemical overalls': 'Macacao Quimico', 
    'seat-belt': 'Cinto de Seguranca',
    'heat': 'Protecao Termica', 
    'without ear protectors': 'Sem Protetor Auricular',
    'without glass': 'Sem Oculos',
    'without glove': 'Sem Luvas',
    'without mask': 'Sem Mascara',
    'without shoes': 'Sem Calcado'
}

CORES_POR_CLASSE = {
    'capacete': (255, 128, 0), 'mascara': (0, 255, 0), 'mascara de solda': (0, 128, 0), 
    'oculos de protecao': (0, 0, 255), 'oculos': (0, 100, 200),       
    'protetor auricular': (255, 255, 0), 'abafador': (200, 200, 0),     
    'colete': (255, 0, 255), 'colete de seguranca': (200, 0, 200), 
    'luvas': (0, 128, 255), 'botas': (128, 0, 128), 'bota de seguranca': (100, 0, 100), 
    'calcado de seguranca': (80, 0, 80), 'calcado': (100, 50, 100),
    'pessoa': (128, 128, 128), 'trabalhador': (160, 160, 160), 
    'sem capacete': (50, 100, 255), 'sem colete': (50, 50, 255),      
    'sem protetor auricular': (100,100,200), 'sem oculos': (0,50,200),
    'sem luvas': (50,200,200), 'sem mascara': (50,200,50), 'sem calcado': (100,50,50),
    'traje de protecao': (0, 128, 128), 'macacao quimico': (0, 100, 100),
    'cinto de seguranca': (128, 128, 0), 'protecao termica': (255, 100, 0)
}
COR_PADRAO_BOX = (100, 100, 100) 

EXCLUIR_DO_MODELO_1 = [] 
EXCLUIR_DO_MODELO_2 = [
    'shoes', 'safety_shoes', 'glasses', 'person', 
    'without glass', 'without shoes', 
    'gloves', 'glove' 
] 

ESPESSURA_CAIXA = 2 
ESCALA_FONTE_LABEL = 0.6
ESPESSURA_FONTE_LABEL = 1

epis_por_pessoa_rastreada = defaultdict(set)

while True:
    ret, frame = cap.read() 
    if not ret or frame is None:
        print("Erro ao capturar frame da câmera. Encerrando.")
        break

    frame_anotado = frame.copy() 
    
    all_detections_xyxy = []
    all_confidences = []
    all_class_ids = []
    all_class_names_original = [] 
    all_class_names_pt = []       
    all_model_source = []         

    if model1:
        results1 = model1(frame, verbose=False, conf=0.1) 
        if results1 and results1[0].boxes: 
            for box in results1[0].boxes:
                confianca = float(box.conf[0])
                if confianca < 0.4: continue
                id_classe = int(box.cls[0])
                nome_classe_original = model1.names.get(id_classe, "DESCONHECIDO_M1")
                
                all_detections_xyxy.append(box.xyxy[0].cpu().numpy())
                all_confidences.append(confianca)
                all_class_ids.append(id_classe) 
                all_class_names_original.append(nome_classe_original)
                all_class_names_pt.append(TRADUCOES_PARA_PORTUGUES.get(nome_classe_original.lower(), nome_classe_original))
                all_model_source.append("M1")

    if model2:
        results2 = model2(frame, verbose=False, conf=0.4) 
        if results2 and results2[0].boxes: 
            for box in results2[0].boxes:
                confianca = float(box.conf[0])
                if confianca < 0.4: continue
                id_classe = int(box.cls[0])
                nome_classe_original = model2.names.get(id_classe, "DESCONHECIDO_M2")

                all_detections_xyxy.append(box.xyxy[0].cpu().numpy())
                all_confidences.append(confianca)
                all_class_ids.append(id_classe) 
                all_class_names_original.append(nome_classe_original)
                all_class_names_pt.append(TRADUCOES_PARA_PORTUGUES.get(nome_classe_original.lower(), nome_classe_original))
                all_model_source.append("M2")

    if all_detections_xyxy:
        detections_all_np = np.array(all_detections_xyxy)
        # confidences_all_np = np.array(all_confidences) # Não é mais necessário criar este separado aqui
        
        person_boxes_np = []
        person_confidences_np = []
        
        epi_boxes_np = []
        epi_confidences_np = [] # <<< INICIALIZADA AQUI
        epi_class_names_original_for_association = [] 
        epi_class_names_pt_for_drawing = []
        epi_model_source_for_drawing = [] 

        for i in range(len(detections_all_np)): # Usar detections_all_np que é o array de caixas
            nome_orig_lower = all_class_names_original[i].lower()
            is_person_m1 = (all_model_source[i] == "M1" and nome_orig_lower in PERSON_CLASSES and nome_orig_lower not in EXCLUIR_DO_MODELO_1)
            is_person_m2 = (all_model_source[i] == "M2" and nome_orig_lower in PERSON_CLASSES and nome_orig_lower not in EXCLUIR_DO_MODELO_2)

            if is_person_m1 or is_person_m2:
                person_boxes_np.append(all_detections_xyxy[i])
                person_confidences_np.append(all_confidences[i])
            else: 
                exclude_epi = False
                if all_model_source[i] == "M1" and nome_orig_lower in EXCLUIR_DO_MODELO_1:
                    exclude_epi = True
                elif all_model_source[i] == "M2" and nome_orig_lower in EXCLUIR_DO_MODELO_2:
                    exclude_epi = True
                
                epi_boxes_np.append(all_detections_xyxy[i])
                epi_confidences_np.append(all_confidences[i]) # <<< POPULADA AQUI
                epi_class_names_original_for_association.append(all_class_names_original[i]) 
                
                if not exclude_epi:
                    epi_class_names_pt_for_drawing.append(all_class_names_pt[i])
                    epi_model_source_for_drawing.append(all_model_source[i]) 
                else: 
                    epi_class_names_pt_for_drawing.append(None) 
                    epi_model_source_for_drawing.append(None)


        tracked_persons_detections = sv.Detections.empty()
        if person_boxes_np:
            person_detections_sv = sv.Detections(
                xyxy=np.array(person_boxes_np),
                confidence=np.array(person_confidences_np),
                class_id=np.array([0] * len(person_boxes_np)) 
            )
            tracked_persons_detections = tracker.update_with_detections(person_detections_sv)

        person_labels_display = [] 
        if len(tracked_persons_detections) > 0 and tracked_persons_detections.tracker_id is not None:
            for _ in tracked_persons_detections.tracker_id: 
                person_labels_display.append("Pessoa") 
            
            frame_anotado = box_annotator.annotate(scene=frame_anotado, detections=tracked_persons_detections)
            frame_anotado = label_annotator.annotate(scene=frame_anotado, detections=tracked_persons_detections, labels=person_labels_display)

        epis_por_pessoa_rastreada_neste_frame = defaultdict(set)
        if epi_boxes_np: 
            # >>> CORREÇÃO AQUI <<<
            # Usa a lista epi_confidences_np que foi populada corretamente
            if epi_confidences_np: # Garante que não está vazia antes de converter para array
                epi_detections_sv_all_candidates = sv.Detections( 
                    xyxy=np.array(epi_boxes_np),
                    confidence=np.array(epi_confidences_np) # <<< USA A LISTA CORRETA
                )

                for i_epi_orig_list in range(len(epi_class_names_original_for_association)): 
                    if epi_class_names_pt_for_drawing[i_epi_orig_list] is None: 
                        continue

                    epi_box = epi_detections_sv_all_candidates.xyxy[i_epi_orig_list] 
                    epi_nome_pt = epi_class_names_pt_for_drawing[i_epi_orig_list]
                    epi_model_src = epi_model_source_for_drawing[i_epi_orig_list]
                    
                    cor_epi = CORES_POR_CLASSE.get(epi_nome_pt.lower().replace("ç", "c").replace("ã", "a").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u"), COR_PADRAO_BOX)
                    label_epi_display = f"{epi_nome_pt}" 

                    x1e, y1e, x2e, y2e = map(int, epi_box)
                    cv2.rectangle(frame_anotado, (x1e, y1e), (x2e, y2e), cor_epi, ESPESSURA_CAIXA)
                    (w_texto, h_texto), baseline = cv2.getTextSize(label_epi_display, cv2.FONT_HERSHEY_SIMPLEX, ESCALA_FONTE_LABEL, ESPESSURA_FONTE_LABEL)
                    pos_fundo_y1 = y1e - h_texto - baseline - 5
                    pos_fundo_y2 = y1e - baseline + 5
                    if pos_fundo_y1 < 0:
                        pos_fundo_y1 = y2e + baseline + 5
                        pos_fundo_y2 = y2e + h_texto + baseline + baseline + 5
                    pos_texto_y = pos_fundo_y1 + h_texto + (baseline // 2)
                    cv2.rectangle(frame_anotado, (x1e, pos_fundo_y1), (x1e + w_texto + 4, pos_fundo_y2), cor_epi, -1)
                    cor_texto_epi = (255,255,255) 
                    cv2.putText(frame_anotado, label_epi_display, (x1e + 2, pos_texto_y), cv2.FONT_HERSHEY_DUPLEX, ESCALA_FONTE_LABEL, cor_texto_epi, ESPESSURA_FONTE_LABEL)

                    if len(tracked_persons_detections) > 0 and tracked_persons_detections.tracker_id is not None:
                        for i_person in range(len(tracked_persons_detections)):
                            person_box = tracked_persons_detections.xyxy[i_person]
                            person_id = tracked_persons_detections.tracker_id[i_person]
                            
                            iou = sv.box_iou_batch(np.array([epi_box]), np.array([person_box]))[0][0]

                            if iou > IOU_THRESHOLD_ASSOCIATION:
                                if "capacete" in epi_nome_pt.lower(): 
                                    if epi_box[1] < (person_box[1] + person_box[3]) / 2: 
                                        epis_por_pessoa_rastreada_neste_frame[person_id].add(epi_class_names_original_for_association[i_epi_orig_list].lower())
                                else: 
                                    epis_por_pessoa_rastreada_neste_frame[person_id].add(epi_class_names_original_for_association[i_epi_orig_list].lower())
            
        epis_por_pessoa_rastreada = epis_por_pessoa_rastreada_neste_frame


    cv2.imshow(WINDOW_NAME, frame_anotado) 

    current_time = time.time()
    if current_time - last_check_time >= 15: 
        print(f"\nVerificando EPIs às {time.strftime('%H:%M:%S')}...")
        if not epis_por_pessoa_rastreada:
            print("  Nenhuma pessoa rastreada para verificar EPIs.")
        
        for person_id, epis_da_pessoa in epis_por_pessoa_rastreada.items():
            print(f"  Pessoa ID {person_id}: EPIs detectados = {epis_da_pessoa}")
            epis_faltando_para_pessoa = False
            for epi_obrigatorio in SAFETY_EPIS_OBRIGATORIOS:
                if epi_obrigatorio.lower() not in epis_da_pessoa:
                    print(f"    FALTANDO para Pessoa ID {person_id}: {epi_obrigatorio}")
                    epis_faltando_para_pessoa = True
            
            if epis_faltando_para_pessoa:
                print(f"  ALERTA: EPIs faltando para Pessoa ID {person_id} detectados!")
            elif epis_da_pessoa: 
                 print(f"  Pessoa ID {person_id}: Todos os EPIs obrigatórios verificados foram detectados.")
        
        last_check_time = current_time
        
    if cv2.waitKey(1) & 0xFF == 27: 
        print("Tecla ESC pressionada, encerrando...")
        break

cap.release()
cv2.destroyAllWindows()
print("Script encerrado.")
