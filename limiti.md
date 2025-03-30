# Appunti sui Limiti - Analisi 1

## Definizione di Limite

Il concetto di limite è fondamentale nell'analisi matematica e descrive il comportamento di una funzione quando la variabile indipendente si avvicina a un particolare valore o tende all'infinito.

### Definizione formale (ε-δ)

Sia $f: D \to \mathbb{R}$ una funzione e sia $c$ un punto di accumulazione per $D$. Diciamo che $\lim_{x \to c} f(x) = L$ se:

$$\forall \varepsilon > 0, \exists \delta > 0 : 0 < |x - c| < \delta \Rightarrow |f(x) - L| < \varepsilon$$

## Proprietà dei Limiti

- **Unicità**: Se esiste il limite di una funzione, esso è unico
- **Linearità**: $\lim_{x \to c} [af(x) + bg(x)] = a\lim_{x \to c} f(x) + b\lim_{x \to c} g(x)$
- **Prodotto**: $\lim_{x \to c} [f(x) \cdot g(x)] = \lim_{x \to c} f(x) \cdot \lim_{x \to c} g(x)$
- **Quoziente**: $\lim_{x \to c} \frac{f(x)}{g(x)} = \frac{\lim_{x \to c} f(x)}{\lim_{x \to c} g(x)}$ se $\lim_{x \to c} g(x) \neq 0$
- **Composizione**: $\lim_{x \to c} f(g(x)) = f(\lim_{x \to c} g(x))$ se $f$ è continua in $\lim_{x \to c} g(x)$

## Limiti Notevoli

- $\lim_{x \to 0} \frac{\sin x}{x} = 1$
- $\lim_{x \to 0} (1 + x)^{\frac{1}{x}} = e$
- $\lim_{x \to 0} \frac{e^x - 1}{x} = 1$
- $\lim_{x \to 0} \frac{\ln(1+x)}{x} = 1$
- $\lim_{x \to 0} \frac{1-\cos x}{x^2} = \frac{1}{2}$

## Limiti da Destra e da Sinistra

- Limite destro: $\lim_{x \to c^+} f(x) = L$
- Limite sinistro: $\lim_{x \to c^-} f(x) = L$
- Il limite esiste se e solo se il limite destro e sinistro coincidono

## Limiti all'Infinito

- $\lim_{x \to \infty} f(x) = L$ significa che i valori di $f(x)$ si avvicinano arbitrariamente a $L$ al crescere di $x$
- $\lim_{x \to -\infty} f(x) = L$ per il comportamento quando $x$ decresce senza limite

## Limiti Infiniti

- $\lim_{x \to c} f(x) = \infty$ significa che $f(x)$ cresce oltre ogni limite quando $x$ si avvicina a $c$
- Analogamente per $\lim_{x \to c} f(x) = -\infty$

## Forme Indeterminate

- $\frac{0}{0}$, $\frac{\infty}{\infty}$, $0 \cdot \infty$, $\infty - \infty$, $0^0$, $\infty^0$, $1^{\infty}$

Per risolvere queste forme si utilizzano tecniche come:

- Regola di De L'Hôpital
- Sviluppi di Taylor
- Scomposizioni algebriche
- Cambi di variabile

## Continuità e Limiti

Una funzione $f$ è continua in $c$ se e solo se:

1. $f(c)$ è definita
2. $\lim_{x \to c} f(x)$ esiste
3. $\lim_{x \to c} f(x) = f(c)$