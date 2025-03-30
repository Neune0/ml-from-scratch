# Appunti sulle Derivate

## Definizione di Derivata

La derivata di una funzione $f(x)$ nel punto $x_0$ è definita come:

$$f'(x_0) = \lim_{h \to 0} \frac{f(x_0 + h) - f(x_0)}{h}$$

Questa definizione cattura l'essenza del calcolo infinitesimale, dove $h$ rappresenta un "delta" infinitamente piccolo. Quando parliamo di "delta piccolissimi", ci riferiamo a incrementi tendenti a zero che ci permettono di analizzare il comportamento della funzione in un intorno infinitesimo del punto.

### Interpretazione Geometrica

Geometricamente, la derivata $f'(x_0)$ rappresenta:

- La pendenza della retta tangente alla curva $y = f(x)$ nel punto $(x_0, f(x_0))$
- Il tasso di variazione istantaneo della funzione in quel punto

Possiamo visualizzare il processo come:

1. Consideriamo due punti sulla curva: $(x_0, f(x_0))$ e $(x_0 + h, f(x_0 + h))$
2. La retta che passa per questi punti è una secante, con pendenza $\frac{f(x_0 + h) - f(x_0)}{h}$
3. Quando $h \to 0$, la secante ruota progressivamente fino a diventare tangente alla curva

Questo passaggio al limite trasforma il tasso di variazione medio in un tasso di variazione istantaneo, fondamentale per comprendere il comportamento locale della funzione.

## Dimostrazioni per le Derivate Fondamentali

### 1. Funzione Costante: $f(x) = c$

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h} = \lim_{h \to 0} \frac{c - c}{h} = \lim_{h \to 0} \frac{0}{h} = 0$$

Quindi: $(c)' = 0$

### 2. Funzione Lineare: $f(x) = x$

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h} = \lim_{h \to 0} \frac{(x + h) - x}{h} = \lim_{h \to 0} \frac{h}{h} = 1$$

Quindi: $(x)' = 1$

### 3. Funzione Potenza: $f(x) = x^n$

$$f'(x) = \lim_{h \to 0} \frac{(x + h)^n - x^n}{h}$$

Utilizzando il binomio di Newton:

$$(x + h)^n = x^n + \binom{n}{1}x^{n-1}h + \binom{n}{2}x^{n-2}h^2 + ... + h^n$$

Sostituendo:

$$f'(x) = \lim_{h \to 0} \frac{x^n + nx^{n-1}h + O(h^2) - x^n}{h} = \lim_{h \to 0} \left(nx^{n-1} + O(h)\right) = nx^{n-1}$$

Quindi: $(x^n)' = nx^{n-1}$

### 4. Funzione Esponenziale: $f(x) = e^x$

$$f'(x) = \lim_{h \to 0} \frac{e^{x+h} - e^x}{h} = \lim_{h \to 0} \frac{e^x \cdot e^h - e^x}{h} = e^x \lim_{h \to 0} \frac{e^h - 1}{h}$$

Sapendo che $\lim_{h \to 0} \frac{e^h - 1}{h} = 1$, otteniamo:

$$f'(x) = e^x \cdot 1 = e^x$$

Quindi: $(e^x)' = e^x$

### 5. Funzione Logaritmica: $f(x) = \ln(x)$

$$f'(x) = \lim_{h \to 0} \frac{\ln(x+h) - \ln(x)}{h} = \lim_{h \to 0} \frac{\ln\left(\frac{x+h}{x}\right)}{h} = \lim_{h \to 0} \frac{\ln\left(1+\frac{h}{x}\right)}{h}$$

Facendo il cambio di variabile $t = \frac{h}{x}$, abbiamo $h = xt$ e $h \to 0 \implies t \to 0$:

$$f'(x) = \lim_{t \to 0} \frac{\ln(1+t)}{xt} = \frac{1}{x} \lim_{t \to 0} \frac{\ln(1+t)}{t}$$

Sapendo che $\lim_{t \to 0} \frac{\ln(1+t)}{t} = 1$, otteniamo:

$$f'(x) = \frac{1}{x}$$

Quindi: $(\ln x)' = \frac{1}{x}$

### 6. Funzione Seno: $f(x) = \sin(x)$

$$f'(x) = \lim_{h \to 0} \frac{\sin(x+h) - \sin(x)}{h}$$

Usando la formula di addizione del seno:

$$f'(x) = \lim_{h \to 0} \frac{\sin x \cos h + \cos x \sin h - \sin x}{h}$$
$$f'(x) = \lim_{h \to 0} \left[\sin x \frac{\cos h - 1}{h} + \cos x \frac{\sin h}{h}\right]$$

Sapendo che $\lim_{h \to 0} \frac{\cos h - 1}{h} = 0$ e $\lim_{h \to 0} \frac{\sin h}{h} = 1$:

$$f'(x) = \sin x \cdot 0 + \cos x \cdot 1 = \cos x$$

Quindi: $(\sin x)' = \cos x$

### 7. Funzione Coseno: $f(x) = \cos(x)$

Con procedimento analogo al seno:

$$(\cos x)' = -\sin x$$