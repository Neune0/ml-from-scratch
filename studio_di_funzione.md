# Studio di Funzione

## Introduzione

Lo studio di funzione è un processo sistematico per analizzare le caratteristiche di una funzione $f: A \rightarrow B$ (come definita nel documento precedente) per comprenderne il comportamento e tracciarne il grafico.

## Fasi dello Studio di Funzione

### 1. Determinazione del Dominio

Il dominio è l'insieme dei valori $x$ per cui la funzione è definita. Si trovano i valori che rendono l'espressione matematica valida, escludendo:

- Denominatori uguali a zero
- Radici di indice pari con argomento negativo
- Logaritmi con argomento non positivo

### 2. Studio della Simmetria

- **Funzione pari**: $f(-x) = f(x)$ (simmetria rispetto all'asse $y$)
- **Funzione dispari**: $f(-x) = -f(x)$ (simmetria rispetto all'origine)

### 3. Intersezioni con gli Assi

- Con l'asse $x$: risolvere $f(x) = 0$
- Con l'asse $y$: calcolare $f(0)$ (se $0$ appartiene al dominio)

### 4. Studio del Segno

Determinare gli intervalli in cui $f(x) > 0$ e quelli in cui $f(x) < 0$.

### 5. Limiti e Asintoti

- **Asintoti verticali**: limiti agli estremi del dominio
- **Asintoti orizzontali**: $\lim_{x \to \pm\infty} f(x) = L$
- **Asintoti obliqui**: $\lim_{x \to \pm\infty} [f(x) - (mx + q)] = 0$

### 6. Derivate e Monotonia

- La derivata prima $f'(x)$ determina gli intervalli di crescenza e decrescenza:
  - Se $f'(x) > 0$, $f(x)$ è crescente
  - Se $f'(x) < 0$, $f(x)$ è decrescente

### 7. Punti Critici

- **Massimi e minimi**: punti in cui $f'(x) = 0$ o $f'(x)$ non esiste
- **Punti di flesso**: punti in cui $f''(x) = 0$ o $f''(x)$ non esiste

### 8. Concavità

Studiare il segno della derivata seconda $f''(x)$:

- Se $f''(x) > 0$, la concavità è verso l'alto
- Se $f''(x) < 0$, la concavità è verso il basso

## Esempio di Studio Completo

Consideriamo la funzione $f(x) = \frac{x^2-1}{x-2}$

1. **Dominio**: $\mathbb{R} \setminus \{2\}$ (tutti i numeri reali tranne 2)
2. **Simmetria**: non è né pari né dispari
3. **Intersezioni**: 
     - Con asse $x$: $x = \pm 1$
     - Con asse $y$: $f(0) = \frac{-1}{-2} = \frac{1}{2}$
4. **Asintoti**:
     - Verticale: $x = 2$
     - Orizzontale: $\lim_{x \to \pm\infty} f(x) = \lim_{x \to \pm\infty} \frac{x^2-1}{x-2} = \lim_{x \to \pm\infty} (x+2+\frac{3}{x-2}) = +\infty$
     - Obliquo: $y = x + 2$

Una rappresentazione grafica seguirebbe tutte queste caratteristiche, fornendo una visualizzazione completa del comportamento della funzione.
