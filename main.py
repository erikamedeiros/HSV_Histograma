import cv2
import numpy as np
import sys
import argparse
from matplotlib import pyplot as plt

#função responsável por extrair as informações do sashibo
def test(location):

    #abrindo imagem
    src = cv2.imread(location)
    #redimensionando a imagem para 600x400
    src = cv2.resize(src, (600, 400))
    #alterando espaço de cores para HSV e jogando a nova imagem na variável hsv
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    #detectando a cor do sashibo por range de cor HSV (vermelho mais escuro até o vermelho mais claro)
    #Obs: Foram realizados vários testes com ranges diferentes de vermelho para se chegar nesses valores abaixo

    #definindo os valores HSV mínimo para detecção de cor
    lower_range = np.array([150, 40, 40])
    #definindo os valores HSV máximos para detecção de cor
    upper_range = np.array([189, 255, 255])

    #criando uma máscara na ára onde não for encontrada a cor que estiver no range definido acima
    mask = cv2.inRange(hsv, lower_range, upper_range)

    #imprimindo a máscara
    #cv2.imshow('mask', mask)

    #foi passado um filtro gausiano para suavizar as bordas da área do sashibo para minimizar as perdas
    gray = cv2.GaussianBlur(mask, (7, 7), 3)

    #aqui convertemos a imagem para bits, onde de o valor do pixel for menos que o limite
    # ele se torna 0 e se for maior se torna 1. Isso ajuda no algoritmo de contornos.
    t, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)

    #contornando a área do sashibo
    contours, a = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #pintando contornos na imagem para visualização
    #cv2.drawContours(src, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)

    #variáveis responsáveis por armazenar os valores da posição do sashibo na imagem original
    x = 0
    y = 0
    w = 0
    h = 0

    #Percorrendo todo contorno para extrair as informações da posição do sashibo
    for c in contours:
        area = cv2.contourArea(c)
        if area > 10 and area < 1000000:
            #criando um retângulo na área encontrada e passando as informações de ponto e área para as variáveis
            (x, y, w, h) = cv2.boundingRect(c)
            #cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
            #print(x,y,w,h)
            break

    #extraindo sashibo
    crop = src[y:y+h, x:x+w]

    #abaixo só trabalhamos com a imagem que foi extraída (que é o sahsibo):

    #padrão usado pelo opencv, vamos usar nas plotagens dos gráficos abaixo
    color = ('b','g','r')

    #plotando gráficos:

    plt.figure(figsize=(6, 4))

    plt.suptitle('Extração de informações do ' + location)

    grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

    plt.subplot(grid[0, 0])
    plt.title('Sashibo')
    plt.imshow(crop)

    plt.subplot(grid[0, 1:])
    plt.title('Histograma referente')

    #calculando histograma para o RGB
    for i,col in enumerate(color):
        histr = cv2.calcHist([crop], [i], None, [256], [0, 256])
        plt.plot(histr, color = col)
        plt.xlim([0, 256])

    plt.subplot(grid[1, :2])
    plt.title('HSV')

    x = np.arange(3)

    #capturando dados HSV da imagem separados
    hue = crop[:, :, 0].mean()
    saturation = crop[:, :, 1].mean()
    value = crop[:, :, 2].mean()

    values = [saturation, hue, value]

    plot = plt.bar(x, values)

    #plotando HSV em gráfico
    for value in plot:
        height = value.get_height()
        plt.text(value.get_x() + value.get_width()/2., 1.002*height,'%d' % int(height), ha='center', va='bottom')

    plt.xticks(x, ('Saturação', 'Matiz', 'Valor'))
    plt.xlabel("hsv separado")
    plt.ylabel("Valor") 

    #cv2.imshow('contornos', src)

    #cv2.imshow('crop', crop)

    #print(hue)
    #print(saturation)

    plt.show(block=False)

    return 0

#função responsável por coletar os argumentos (nome dos aquivos de imagem) passados na chamada do programa 
def main():
    parser = argparse.ArgumentParser(description='Imagens a serem processadas')
    
    #aqui verifica se a flag -l foi passada
    parser.add_argument('-l','--list', nargs='+', help='<Required> Set flag', required=True)

    #pegando os nomes de arquivos passados e colocando na variável value
    for _, value in parser.parse_args()._get_kwargs():
        if value is not None:
            break

    print('Processando imagens...')

    #loop que roda a função test para cada arquivo de imagem
    for n in value:
        test(n)

    print('<<< Processado >>>')

    #pausando aplicação para manter as janelas dos gráficos abertas
    while True:
        key = cv2.waitKey(1)
        #se clicar ESC o programa fecha (caso n feche é só encerrar o processo pelo terminal (Control + C))
        if key == 27:
            cv2.destroyAllWindows()
    return 0

#chamada da função main
if __name__ == '__main__':
    sys.exit(main())