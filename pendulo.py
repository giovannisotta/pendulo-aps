import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

# Índices das linhas do array "dados" (cada linha é uma grandeza, cada coluna é um frame)
X, Y, T, R = 0, 1, 2, 3
FILE_PATH   = "pendulo.mp4"
WINDOW_NAME = "pendulo"

# Faixa de cor da laranja no espaço HSV: (matiz, saturação, valor)
LOWER   = np.array([5,100,80])
UPPER   = np.array([30,255,255])

ESCALA = 0.5                    # reduz o frame pela metade para processar mais rápido
K      = int(20 * ESCALA)       # tamanho do kernel morfológico, proporcional à escala
X0, X1 = 0, None                # recorte horizontal do frame (None = até o fim)
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K, K))   # kernel elíptico para limpar a máscara

D_REAL = 0.09                   # diâmetro real da laranja, em metros (usado na calibração)

EXIBIR = True                   # True mostra a janela com o tracking (mais lento)

# ---------------------------------------------------------------------------
# Extração: vídeo -> números
# ---------------------------------------------------------------------------

def extrair(path):
    """Lê o vídeo e devolve (dados, r_med).

    dados: array (3, N) com as linhas x, y e t (pixels e segundos), um frame por coluna.
    r_med: mediana do raio da laranja em pixels (usada na calibração).
    """
    dados: list[list[float]] = [[], [], []]   # listas de x, y e t, preenchidas em paralelo
    r_list: list[float] = []                  # raio de cada frame
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_i = 0;
    while True:
        ok, frame = cap.read()
        if not ok: break                      # acabou o vídeo
        t = frame_i / fps                     # tempo do frame = número do frame / fps
        frame_i += 1
        frame = frame[:, X0:X1]               # recorte horizontal
        frame = cv2.resize(frame, None, fx=ESCALA, fy=ESCALA)

        # Isola a laranja por cor: BGR -> HSV, filtro de faixa e fechamento morfológico
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER, UPPER)                          # 255 onde é laranja, 0 no resto
        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)   # tapa buracos pequenos na máscara

        # Acha os contornos e fica com o maior (a laranja)
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue             # frame sem laranja detectada: pula

        c = max(contours, key=cv2.contourArea)
        m = cv2.moments(c)                    # momentos do contorno (m00 = área)
        if m["m00"] == 0: continue

        cx = m["m10"] / m["m00"]              # centroide em x
        cy = m["m01"] / m["m00"]              # centroide em y
        r  = np.sqrt(m["m00"] / np.pi)        # raio equivalente de um círculo com a mesma área

        # Os quatro append ficam juntos: se o frame foi pulado, nenhuma lista cresce
        dados[X].append(cx)
        dados[Y].append(cy)
        dados[T].append(t)
        r_list.append(r)

        if EXIBIR:
            _ = cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)        # contorno verde
            _ = cv2.circle(frame, (int(cx), int(cy)), 10, (0, 0, 255), -1)   # centroide vermelho
            cv2.imshow("frame", frame)
            if cv2.waitKey(33) == ord("q"): break    # ~30 fps; "q" interrompe
    cap.release()
    cv2.destroyAllWindows()
    return np.array(dados), np.median(r_list)

# ---------------------------------------------------------------------------
# Preparação e calibração
# ---------------------------------------------------------------------------

def preparar(dados, t_inicio=4):
    # Descarta os primeiros segundos (a mão soltando a laranja).
    # Filtra as colunas (frames), então x, y e t continuam alinhados.
    return dados[:, dados[T] > t_inicio]

def calibrar(r_med, d_real):
    # Metros por pixel: diâmetro real / diâmetro em pixels (2 * raio)
    return d_real / (2 * r_med)

# ---------------------------------------------------------------------------
# Modelo e ajuste
# ---------------------------------------------------------------------------

def modelo(t, x0, A, b, w, phi):
    # Oscilador harmônico amortecido (amortecimento fraco):
    # x0 = centro, A = amplitude inicial, b = taxa de queda da amplitude,
    # w = frequência angular, phi = fase
    return x0 + A * np.exp(-b*t) * np.cos(w * t + phi)

def ajustar(t, x):
    t = t - t[0]                              # o modelo espera o tempo começando em 0
    # Chute inicial: centro = média, amplitude = metade da faixa, período ~1,75 s
    p0 = [x.mean(), (x.max() - x.min()) / 2, 0.01, 2 * np.pi / 1.75, 0]
    popt, pcov = curve_fit(modelo, t, x, p0=p0)   # popt = parâmetros, pcov = covariância
    erro = np.sqrt(np.diag(pcov))                  # incerteza de cada parâmetro
    return popt, erro

# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dados, r_med = extrair(FILE_PATH)
    dados = preparar(dados)
    m_por_px = calibrar(r_med, D_REAL)

    t = dados[T] - dados[T][0]                # tempo começando em 0
    x = dados[X] * m_por_px                   # posição em metros

    popt, erro = ajustar(dados[T], x)
    x0, A, b, w, phi = popt

    periodo = 2 * np.pi / w                   # T = 2π/ω
    Q = w / (2 * b)                           # fator de qualidade
    print(f"x0  = {x0:.5f}  ± {erro[0]:.5f} m")
    print(f"A   = {A:.5f}   ± {erro[1]:.5f} m")
    print(f"b   = {b:.5f}   ± {erro[2]:.5f} 1/s")
    print(f"w   = {w:.5f}   ± {erro[3]:.5f} rad/s")
    print(f"phi = {phi:.4f} ± {erro[4]:.4f} rad")    
    print(f"T   = {periodo:.5f} s")
    print(f"Q   = {Q:.1f}")

    # Resíduo = medido - ajustado (deve ser ruído em torno de zero se o modelo é bom)
    res = x - modelo(t, *popt)

    # Tabela com uma linha por frame: t, x, y, x ajustado e resíduo
    tabela = np.column_stack((
        dados[T],
        dados[X] * m_por_px,
        dados[Y] * m_por_px,
        modelo(t, *popt),
        res
    ))
    with open("tabela.csv", "w") as f:
        # Parâmetros do ajuste como linhas de comentário (#) no topo
        f.write(f"# T   = {periodo:.5f} s\n")
        f.write(f"# b   = {b:.5f} +- {erro[2]:.5f} 1/s\n")
        f.write(f"# w   = {w:.5f} rad/s\n")
        f.write(f"# A   = {A:.6f} m\n")
        f.write(f"# phi = {phi:.5f} rad\n")
        f.write(f"# x0  = {x0:.6f} m\n")
        f.write(f"# Q   = {Q:.1f}\n")
        f.write("t (s),x (m),y (m),x_ajuste (m),residuo (m)\n")   # cabeçalho das colunas
        np.savetxt(f, tabela, delimiter=",")

    # Gráfico: dados com a curva ajustada em cima, e resíduos embaixo
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t, x, ".")
    ax1.plot(t, modelo(t, *popt), "r-")
    ax1.set_ylabel("x (m)")
    ax2.plot(t, res, ".")
    ax2.set_ylabel("resíduo (m)")
    ax2.set_xlabel("t (s)")
    plt.show()
