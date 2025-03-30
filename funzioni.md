# Funzioni

## Definizione di Funzione

Una **funzione** è una relazione tra due insiemi, diciamo $A$ (dominio) e $B$ (codominio), in cui ad ogni elemento del dominio $A$ viene associato esattamente un elemento del codominio $B$.

Formalmente, una funzione $f: A \rightarrow B$ è un sottoinsieme del prodotto cartesiano $A \times B$ tale che:

- Per ogni $a \in A$, esiste un elemento $b \in B$ tale che $(a, b) \in f$
- Se $(a, b_1) \in f$ e $(a, b_2) \in f$, allora $b_1 = b_2$

## Controesempio: Relazione che Non è una Funzione

Consideriamo la relazione $R$ da $A = \{1, 2, 3\}$ a $B = \{a, b, c\}$ definita come:
$R = \{(1, a), (1, b), (2, b), (3, c)\}$

Questa non è una funzione perché l'elemento $1 \in A$ è associato a due elementi diversi del codominio: $a$ e $b$. Una funzione deve associare ogni elemento del dominio ad esattamente un elemento del codominio.

## Proprietà delle Funzioni

### Iniettività

Una funzione $f: A \rightarrow B$ si dice **iniettiva** (o "uno-a-uno") se:

- Per ogni $a_1, a_2 \in A$, se $a_1 \neq a_2$ allora $f(a_1) \neq f(a_2)$

In altre parole, elementi diversi nel dominio hanno immagini diverse nel codominio.

Esempio: $f: \mathbb{R} \rightarrow \mathbb{R}$ definita da $f(x) = 2x + 1$ è iniettiva.

### Suriettività

Una funzione $f: A \rightarrow B$ si dice **suriettiva** (o "onto") se:

- Per ogni $b \in B$ esiste almeno un $a \in A$ tale che $f(a) = b$

In altre parole, ogni elemento del codominio è l'immagine di almeno un elemento del dominio.

Esempio: $f: \mathbb{R} \rightarrow \mathbb{R}$ definita da $f(x) = e^x$ non è suriettiva perché non esistono valori di $x$ tali che $f(x) < 0$.

### Biettività

Una funzione $f: A \rightarrow B$ si dice **biettiva** (o "biunivoca") se è sia iniettiva che suriettiva.

Proprietà di una funzione biettiva:

- Ogni elemento di $B$ è l'immagine di esattamente un elemento di $A$
- Esiste la funzione inversa $f^{-1}: B \rightarrow A$

Esempio: $f: \mathbb{R} \rightarrow \mathbb{R}$ definita da $f(x) = 3x - 2$ è biettiva.
