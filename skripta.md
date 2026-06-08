# ISPITNA SKRIPTA - Statistička analiza podataka
---


# SADRŽAJ

1. [Višestruka linearna regresija](#1-višestruka-linearna-regresija)
2. [Dijagnostika reziduala](#2-dijagnostika-reziduala)
3. [Kolinearnost i VIF](#3-kolinearnost-i-vif)
4. [Kvalitativni prediktori (faktori)](#4-kvalitativni-prediktori-i-interakcije)
5. [Odabir modela: R², AIC, prilagođeni R²](#5-odabir-modela)
6. [Logistička regresija](#6-logistička-regresija)
7. [Mjere kvalitete klasifikacije](#7-mjere-kvalitete-klasifikacije)
8. [KNN klasifikacija](#8-knn-klasifikacija)
9. [Generalizirani linearni modeli – Poissonova regresija](#9-poissonova-regresija)
10. [Cross-validacija (CV)](#10-cross-validacija)
11. [Bootstrap](#11-bootstrap)
12. [Stabla odluke (Decision Trees)](#12-stabla-odluke)
13. [Random Forest](#13-random-forest)
14. [Support Vector Machine (SVM)](#14-support-vector-machine-svm)
15. [PCA – Analiza glavnih komponenti](#15-pca--analiza-glavnih-komponenti)
16. [Završni globalni sažetak i checklist](#16-završni-globalni-sažetak)

---

# 1. Višestruka linearna regresija

## Teorija

Kada imamo **kvantitativnu odzivnu varijablu** Y i više prediktora X₁, X₂, ..., Xₚ, koristimo višestruku linearnu regresiju. Model kaže da je Y linearna kombinacija prediktora plus slučajna greška:

```
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε
```

Koeficijent βᵢ znači: **za koliko se Y promijeni ako Xᵢ porastu za 1, uz sve ostale prediktore nepromijenjene.** To je ključna razlika od jednostavne regresije.

Parametri se procjenjuju **metodom najmanjih kvadrata** (minimizacija RSS).

**Kada koristiti:** Kada odzivna varijabla Y je kontinuirana i pretpostavljamo linearnu vezu s prediktorima.

## Ključni pojmovi

| Pojam | Značenje |
|---|---|
| **RSS** | Suma kvadrata reziduala – mjera neprilagodbe modela |
| **MSE** | Srednja kvadratna greška = RSS/n |
| **R²** | Udio objašnjene varijance (0–1, veći = bolji) |
| **Prilagođeni R²** | R² korigiran za broj prediktora (penalizira nepotrebne prediktore) |
| **p-vrijednost** | Vjerojatnost da bismo dobili ovaj efekt ako β=0; mala p → značajan prediktor |
| **Reziduali** | Razlika između stvarnih i procijenjenih vrijednosti: eᵢ = yᵢ - ŷᵢ |
| **Trening MSE** | MSE na podacima za učenje (optimistična procjena) |
| **Testni MSE** | MSE na novim podacima (realna procjena generalizacije) |
| **Bias-variance tradeoff** | Fleksibilniji modeli imaju manju pristranost ali veću varijancu |

## R kodovi

### Osnovna analiza

```r
# Učitavanje paketa i podataka
library(ISLR2)
data(Auto)

# Pregled podataka – uvijek prvi korak!
str(Auto)
summary(Auto)
head(Auto)

# Korelacijska matrica – pronalazi linearne veze
cor(Auto[, sapply(Auto, is.numeric)])

# Para (scatter plot matrix)
pairs(Auto[, c("mpg", "horsepower", "weight", "displacement")])
```

### Izgradnja i pregled modela

```r
# Jednostavna linearna regresija
mod1 <- lm(mpg ~ horsepower, data = Auto)
summary(mod1)

# Višestruka linearna regresija
mod2 <- lm(mpg ~ horsepower + weight + displacement, data = Auto)
summary(mod2)

# Što znači summary output:
# Coefficients: procijenjeni βᵢ, Std. Error, t-vrijednost, p-vrijednost
# Residual standard error: procjena σ (SD greške)
# R-squared: udio objašnjene varijance
# Adjusted R-squared: korigiran za broj prediktora
# F-statistic: test je li model kao cjelina značajan

# Procijenjeni koeficijenti
coef(mod2)

# Predikcija na novim podacima
novi <- data.frame(horsepower = 100, weight = 3000, displacement = 200)
predict(mod2, newdata = novi)

# Interval predikcije (za jedno opažanje) – širi
predict(mod2, newdata = novi, interval = "prediction")

# Interval pouzdanosti (za srednju vrijednost) – uži
predict(mod2, newdata = novi, interval = "confidence")
```

### Transformacije i polinomi

```r
# Kvadratni model (nelinearna veza)
mod_kv <- lm(mpg ~ horsepower + I(horsepower^2), data = Auto)
summary(mod_kv)

# Logaritamska transformacija odzivne varijable
mod_log <- lm(log(mpg) ~ horsepower, data = Auto)

# Interakcija između prediktora
mod_int <- lm(mpg ~ horsepower * weight, data = Auto)
# Ekvivalentno: mpg ~ horsepower + weight + horsepower:weight
```

### Usporedba modela

```r
# AIC – manji je bolji (penalizira kompleksnost)
AIC(mod1, mod2, mod_kv)

# BIC
BIC(mod1, mod2)

# F-test usporedba ugniježđenih modela
anova(mod1, mod2)
# Ako je p < 0.05 → složeniji model je značajno bolji
```

## Interpretacija outputa

**Primjer interpretacije koeficijenta:**
> "Koeficijent uz varijablu `horsepower` iznosi -0.158 i statistički je značajan (p < 0.001). To znači da povećanje snage motora za 1 KS, uz sve ostale prediktore nepromijenjene, smanjuje potrošnju goriva za 0.158 milja po galoni."

**Primjer interpretacije R²:**
> "Model objašnjava 74.8% varijabilnosti varijable `mpg`, što upućuje na solidnu prilagodbu podataka."

**Primjer interpretacije F-statistike:**
> "F-statistika iznosi 234.5 uz p-vrijednost < 2.2e-16, što znači da je model kao cjelina statistički značajan, tj. barem jedan od prediktora ima značajan doprinos objašnjenju varijable `mpg`."

**Primjer interpretacije ANOVA usporedbe:**
> "F-test usporedbe linearnog i kvadratnog modela daje p < 0.001, što znači da kvadratni model statistički značajno bolje opisuje podatke od linearnog."

## Najčešće greške i zamke

- **Ne provjeriš dijagnostiku** – visok R² ne znači automatski dobar model! Uvijek pogledaj dijagnostičke grafove.
- **Miješaš trening i testnu pogrešku** – smanjivanje trening MSE dodavanjem prediktora ne znači bolji model na novim podacima.
- **Zaboraviš na multikolinearnost** – ako dva prediktora jako koreliraju, koeficijenti postaju nestabilni.
- **Tumačiš βᵢ kao u jednostavnoj regresiji** – u višestrukoj regresiji βᵢ je uvjetni efekt (uz ostale prediktore fiksirane).
- **R² uvijek raste s dodavanjem prediktora** – zato gledaj prilagođeni R².

## Mini ispitni podsjetnik – Linearna regresija

| Što trebam | R kod |
|---|---|
| Izgraditi model | `lm(Y ~ X1 + X2, data = df)` |
| Pregled modela | `summary(model)` |
| Usporediti modele | `AIC(m1, m2)` ili `anova(m1, m2)` |
| Predvidjeti | `predict(model, newdata = novi_df)` |
| Dijagnostika | `par(mfrow=c(2,2)); plot(model)` |

**Zaključak za ispit:**
> "Model višestruke linearne regresije s prediktorima X1 i X2 postiže prilagođeni R² = ... što znači da prediktori zajedno objašnjavaju ...% varijabilnosti odzivne varijable. Koeficijenti su statistički značajni (p < 0.05). Dijagnostički grafovi ne ukazuju na ozbiljna kršenja pretpostavki."

---

# 2. Dijagnostika reziduala

## Teorija

Pretpostavke linearne regresije koje provjeravamo dijagnostikom:
1. **Linearnost** – veza između Y i X je linearna
2. **Normalnost reziduala** – ε ~ N(0, σ²)
3. **Homoskedastičnost** – varijanca greške je konstantna
4. **Nezavisnost reziduala** – nema autokorelacije

Dijagnostiku radimo grafički (4 grafa) i testovima.

## R kodovi

### Dijagnostički grafovi

```r
# 4 dijagnostička grafa odjednom
par(mfrow = c(2, 2))
plot(model)

# Graf 1: Residuals vs Fitted
# → Trebamo nasumičan raspored oko 0 (bez uzorka)
# → Crvena linija treba biti ravna (linearna veza OK)
# → Lijevkast oblik = heteroskedastičnost

# Graf 2: Q-Q plot reziduala
# → Točke trebaju ležati na pravcu
# → Devijacije u repovima = nenormalnost

# Graf 3: Scale-Location (sqrt|reziduali| vs fitted)
# → Crvena linija treba biti ravna
# → Rastući trend = heteroskedastičnost

# Graf 4: Residuals vs Leverage
# → Identificira utjecajna opažanja (Cook's distance)
# → Opažanja izvan Cookove distance od interesa

# Histogram reziduala
hist(residuals(model), breaks = 30, col = "lightskyblue2",
     main = "Histogram reziduala")

# Scatter plot reziduala
plot(fitted(model), residuals(model),
     xlab = "Fitted values", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)
```

### Testovi normalnosti i homoskedastičnosti

```r
# Shapiro-Wilkov test normalnosti reziduala
shapiro.test(residuals(model))
# H₀: reziduali su normalno distribuirani
# p < 0.05 → odbacujemo normalnost

# Breusch-Pagan test homoskedastičnosti
# install.packages("lmtest")
library(lmtest)
bptest(model)
# H₀: homoskedastičnost (varijanca je konstantna)
# p < 0.05 → heteroskedastičnost (problem!)
```

### Simulacijski primjer iz materijala

```r
# Homoskedastični primjer (iz materijala Brborović)
set.seed(42)
n <- 150
x <- runif(n, 0, 10)
y1 <- 2 + 3*x + rnorm(n, mean = 0, sd = 1)  # konstantna σ
modsim1 <- lm(y1 ~ x)

par(mfrow = c(2, 2))
plot(modsim1)
bptest(modsim1)  # Očekivamo p > 0.05

# Heteroskedastični primjer (iz materijala Brborović)
sigma <- 0.3 + 0.3*x          # varijanca raste s x!
y2 <- 2 + 3*x + rnorm(n, mean = 0, sd = sigma)
modsim2 <- lm(y2 ~ x)

par(mfrow = c(2, 2))
plot(modsim2)
bptest(modsim2)  # Očekujemo p < 0.05
```

### Q-Q plot simulacije (iz materijala)

```r
# Normalni podaci
set.seed(42)
x_norm <- rnorm(150, mean = 5, sd = 2)
qqnorm(x_norm)
qqline(x_norm, col = "red")

# t-razdioba (teži repovi)
x_t <- rt(150, df = 4)
qqnorm(x_t)
qqline(x_t, col = "red")
# Na Q-Q plotu: točke odmiču od pravca u repovima (S-oblik)

# Uniformna razdioba
x_unif <- runif(150, -3, 3)
qqnorm(x_unif)
qqline(x_unif, col = "red")
# Na Q-Q plotu: točke se savijaju na oba kraja (obrnuti S)
```

## Interpretacija grafova

**Residuals vs Fitted:**
> "Na grafu 'Residuals vs Fitted' reziduali su nasumično raspoređeni oko nule bez vidljivog uzorka, a crvena linija je ravna. To upućuje da je pretpostavka linearnosti zadovoljena i da nema heteroskedastičnosti."

**Q-Q plot:**
> "Na Q-Q plotu reziduali gotovo u potpunosti prate dijagonalni pravac, što upućuje na normalnost raspodjele reziduala. Mala odstupanja u repovima su prihvatljiva."

**Breusch-Pagan test:**
> "Breusch-Pagan test daje p-vrijednost = 0.32 (> 0.05), što znači da ne možemo odbaciti hipotezu o homoskedastičnosti. Varijanca reziduala je konstantna."

**Heteroskedastičnost:**
> "Na grafu 'Scale-Location' vidimo rastuću crvenu liniju, što upućuje na heteroskedastičnost. Breusch-Pagan test potvrđuje ovaj nalaz (p < 0.001). Standardne pogreške procijenjenih koeficijenata mogu biti podcijenjene, što utječe na zaključke o statističkoj značajnosti."

## Najčešće greške

- **Ne pogledaš Q-Q plot** – visok R² ne jamči normalnost reziduala.
- **Zaboraviš `par(mfrow = c(2,2))`** – bez toga grafovi se prikazuju odvojeno.
- **Krivo interpretiraš BP test** – p > 0.05 znači da NE ODBACUJEMO homoskedastičnost (dakle, homoskedastičnost je OK).

## Mini ispitni podsjetnik – Dijagnostika

| Što testiram | Metoda | R kod |
|---|---|---|
| Normalnost reziduala | Shapiro-Wilk | `shapiro.test(residuals(m))` |
| Homoskedastičnost | Breusch-Pagan | `bptest(m)` |
| Grafička provjera | 4 dijagnostička grafa | `plot(m)` |
| Q-Q plot | Vizualni | `qqnorm(); qqline()` |

---

# 3. Kolinearnost i VIF

## Teorija

**Kolinearnost** (multikolinearnost) znači da su dva ili više prediktora međusobno jako korelirani. Problem: koeficijenti postaju nestabilni i teško je interpretirati individualne doprinose. Model može imati visok R², ali koeficijenti mogu biti statistički neznačajni (visoke p-vrijednosti) i mijenjaju se drastično ako dodamo/uklonimo varijablu.

**VIF (Variance Inflation Factor)** mjeri koliko se varijanca procijenjenog koeficijenta povećava zbog kolinearnosti.

- VIF = 1: nema kolinearnosti
- VIF > 5: umjerena kolinearnost (pazi)
- VIF > 10: ozbiljna kolinearnost (problem!)

## R kodovi

```r
# VIF iz paketa car
# install.packages("car")
library(car)

mod_credit <- lm(Balance ~ Income + Limit + Rating + Cards + Age + Student,
                 data = Credit)
vif(mod_credit)

# Korelacijska matrica numeričkih varijabli
library(ISLR2)
data(Credit)
cor(Credit[, c("Balance", "Income", "Limit", "Rating", "Cards", "Age")])

# Ako je korelacija između dvaju prediktora > 0.8 → kolinearnost!
```

### Primjer iz materijala – Limit vs Rating

```r
# Oba prediktora posebno su značajni
modela <- lm(Balance ~ Limit, data = Credit)
modelb <- lm(Balance ~ Rating, data = Credit)
summary(modela)
summary(modelb)

# Zajedno postaju neznačajni zbog kolinearnosti!
modelc <- lm(Balance ~ Limit + Rating, data = Credit)
summary(modelc)
# ← Uočiti: oba postanu neznačajna, ali R² ostaje visok

# Rješenje: ukloniti jedan od kolinearnih prediktora
model_bez_rating <- lm(Balance ~ . -Rating, data = Credit)
summary(model_bez_rating)
```

## Interpretacija VIF-a

> "Prediktor `Limit` ima VIF = 160.7, što ukazuje na ozbiljnu kolinearnost s ostalim prediktorima. Prediktor `Rating` ima VIF = 153.1. Oba prediktora su međusobno visoko korelirana (r ≈ 0.997), pa ih ne treba koristiti zajedno u modelu. Preporučuje se ukloniti jedan od njih."

## Mini podsjetnik – VIF

| Vrijednost VIF | Interpretacija | Akcija |
|---|---|---|
| 1 – 5 | OK | Ništa |
| 5 – 10 | Pazi | Razmotriti |
| > 10 | Problem | Ukloniti jedan prediktor |

---

# 4. Kvalitativni prediktori i interakcije

## Teorija

Kvalitativni prediktori (faktori) u linearnoj regresiji se automatski kodiraju kao **dummy (indikatorske) varijable**. Za faktor s K razina R kreira K-1 dummy varijabli (jedna razina je referentna). Koeficijent uz dummy varijablu interpretira se kao razlika u Y u odnosu na referentnu kategoriju.

**Interakcija** između prediktora znači da efekt jednog prediktora ovisi o vrijednosti drugog. U R-u se označava s `*` ili `:`.

## R kodovi

```r
library(ISLR2)
data(Credit)

# Provjera razina faktora
levels(Credit$Student)  # "No", "Yes"
contrasts(Credit$Student)  # prikazuje dummy kodiranje

# Model s kvalitativnim prediktorom (bez interakcije)
# Dva paralelna pravca (isti nagib, različiti odsječci)
model_student <- lm(Balance ~ Income + Student, data = Credit)
summary(model_student)
# Koeficijent uz StudentYes = razlika u Balance između studenata i ne-studenata
# uz isti prihod

# Model s interakcijom (različiti nagibi I odsječci)
model_int <- lm(Balance ~ Income * Student, data = Credit)
summary(model_int)
# Income:StudentYes = razlika u nagibu za studente vs ne-studente
```

### Faktor s više razina

```r
data(Carseats)
str(Carseats)  # ShelveLoc ima 3 razine: Bad, Good, Medium

model_shelf <- lm(Sales ~ Price + ShelveLoc, data = Carseats)
summary(model_shelf)
# R automatski kreira: ShelveLocGood i ShelveLocMedium
# Referentna kategorija: Bad

# Interpretacija: ShelveLocGood = 4.84 znači da s dobrom lokacijom
# prodaja je za 4.84 jedinice viša u odnosu na lošu lokaciju
```

### Vizualizacija interakcije

```r
# Boxplot za provjeru efekta kvalitativne varijable
boxplot(Balance ~ Student, data = Credit, col = "lightskyblue2",
        main = "Balance prema statusu studenta")

# Scatter plot s bojama prema razinama faktora
plot(Credit$Income, Credit$Balance,
     col = ifelse(Credit$Student == "Yes", "red3", "blue"),
     pch = 16,
     main = "Income vs Balance (po student statusu)")
legend("topright", legend = c("Ne-student", "Student"),
       col = c("blue", "red3"), pch = 16)
```

## Interpretacija koeficijenata za dummy varijable

> "Koeficijent uz `StudentYes` iznosi 382.7 i statistički je značajan (p < 0.001). To znači da studenti, uz isti prihod, imaju prosječno 382.7 dolara veće stanje na kreditnoj kartici od ne-studenata."

> "U modelu s interakcijom `Income:StudentYes`, koeficijent = -11.97 znači da je nagib veze između prihoda i balansa za studente za 11.97 manji nego za ne-studente. Efekt prihoda na balance je različit ovisno o studentskom statusu."

---

# 5. Odabir modela

## Teorija

Kriteriji za usporedbu i odabir modela:

- **R²** – udio objašnjene varijance; uvijek raste s dodavanjem prediktora → NIJE pouzdan za usporedbu
- **Prilagođeni R²** – penalizira nepotrebne prediktore; može se smanjiti
- **AIC** (Akaike Information Criterion) – manji = bolji; penalizira kompleksnost
- **BIC** – slično AIC-u, ali jača penalizacija
- **RSS** – apsolutna suma kvadrata; manji = bolji fit (ali ne uspoređuj modele različitih dimenzija)

## R kodovi

```r
# Usporedba tri modela (iz materijala Brborović - Auto dataset)
library(ISLR2)
data(Auto)

mod1 <- lm(mpg ~ horsepower, data = Auto)
mod2 <- lm(mpg ~ horsepower + I(horsepower^2), data = Auto)
mod3 <- lm(mpg ~ horsepower + I(horsepower^2) + I(horsepower^3), data = Auto)

# AIC i BIC
AIC(mod1, mod2, mod3)
BIC(mod1, mod2, mod3)

# Prilagođeni R²
summary(mod1)$adj.r.squared
summary(mod2)$adj.r.squared
summary(mod3)$adj.r.squared

# F-test za ugniježđene modele
anova(mod1, mod2)      # Linearni vs kvadratni
anova(mod2, mod3)      # Kvadratni vs kubični

# Prikazivanje svih R² odjednom
sapply(list(mod1, mod2, mod3), function(m) summary(m)$r.squared)
sapply(list(mod1, mod2, mod3), function(m) summary(m)$adj.r.squared)
```

## Interpretacija

> "Kvadratni model ima AIC = 1971.0, što je niže od linearnog modela (AIC = 2089.8) i kubičnog (AIC = 1972.4). Prema AIC kriteriju, kvadratni model je preferiran."

> "Prilagođeni R² kvadratnog modela (0.688) je viši od linearnog (0.605) ali gotovo jednak kubičnom (0.686). F-test između linearnog i kvadratnog modela daje p < 0.001 – kvadratni je statistički značajno bolji. F-test između kvadratnog i kubičnog daje p = 0.24 – kubični ne donosi značajno poboljšanje."

---

# 6. Logistička regresija

## Teorija

Koristimo logističku regresiju kada je odzivna varijabla **binarna** (0/1, Da/Ne). Problem s linearnom regresijom za binarne odzive: može davati vrijednosti izvan [0,1].

Rješenje: **logit transformacija** – modeliramo logaritam izgleda (log-odds):

```
log(p/(1-p)) = β₀ + β₁X₁ + ... + βₚXₚ
```

gdje je p = P(Y=1 | X). Iz toga dobivamo:

```
p(x) = e^(β₀+β₁x) / (1 + e^(β₀+β₁x))    ← logistička funkcija (S-krivulja)
```

**Izgledi (odds)** = p/(1-p). Ako eksponenciramo koeficijent → **omjer izgleda** (odds ratio).

Koeficijenti se procjenjuju **metodom maksimalne vjerodostojnosti** (ne MNK!).

## Ključni pojmovi

| Pojam | Značenje |
|---|---|
| **logit(p)** | log(p/(1-p)) – može biti bilo koji realan broj |
| **odds** | p/(1-p) – izgledi da se Y=1 dogodi |
| **odds ratio** | exp(βᵢ) – koliko puta se mijenjaju izgledi za 1 jedinicu Xᵢ |
| **Null deviance** | "loša" mjera modela bez prediktora – samo intercept |
| **Residual deviance** | loša mjera modela s prediktorima; manja = bolji model |
| **AIC** | Manji = bolji model (za usporedbu logističkih modela) |

## R kodovi

### Osnovna logistička regresija (primjer Default iz materijala)

```r
library(ISLR2)
data(Default)
str(Default)
summary(Default)
table(Default$default)   # Provjeri raspodjelu klasa

# Boxplot za istraživanje
par(mfrow = c(1, 2))
boxplot(income ~ default, data = Default, col = "lightskyblue2",
        main = "Prihod vs default")
boxplot(balance ~ default, data = Default, col = "lightskyblue2",
        main = "Balance vs default")

# Logistička regresija s jednim prediktorom
mod_log1 <- glm(default ~ balance, data = Default, family = binomial)
summary(mod_log1)
```

### Interpretacija koeficijenata

```r
# Koeficijenti na logit-skali
coef(mod_log1)

# Omjer izgleda (odds ratio) – eksponenciranje
exp(coef(mod_log1))
# exp(β₁) = 1.006 → porast balance za 1$ množi izglede s 1.006
# Pregledno: porast za 100$ množi izglede s exp(100 * β₁)
exp(100 * coef(mod_log1)["balance"])

# Interval pouzdanosti za odds ratios
exp(confint(mod_log1))
```

### Procjena vjerojatnosti i klasifikacija

```r
# Procijenjene vjerojatnosti (P(default=Yes | x))
vjerojatnoci <- predict(mod_log1, type = "response")
head(vjerojatnoci)

# Klasifikacija s pragom 0.5
predikcija <- ifelse(vjerojatnoci > 0.5, "Yes", "No")

# Klasifikacijska tablica (matrica zabune / confusion matrix)
table(Predikcija = predikcija, Stvarno = Default$default)

# Točnost
mean(predikcija == Default$default)

# Vizualizacija logističke krivulje
xseq <- seq(0, 2654, length.out = 300)
pr_krivulja <- predict(mod_log1,
                        newdata = data.frame(balance = xseq),
                        type = "response")
plot(Default$balance, as.numeric(Default$default) - 1,
     pch = 16, col = "gray50",
     xlab = "Balance", ylab = "P(default = Yes)")
lines(xseq, pr_krivulja, col = "blue3", lwd = 2)
```

### Višestruka logistička regresija

```r
# Model s više prediktora
mod_log2 <- glm(default ~ balance + income + student,
                data = Default, family = binomial)
summary(mod_log2)

# Usporedba modela (manji AIC = bolji)
AIC(mod_log1, mod_log2)

# Provjera modelom objašnjene devijance
mod_log2$null.deviance    # devijanca bez prediktora
mod_log2$deviance         # devijanca s prediktorima
# Razlika → koliko prediktori poboljšavaju model
```

### Trening / testni skup i evaluacija

```r
set.seed(123)
n <- nrow(Default)
train_id <- sample(1:n, size = round(0.7*n))
train <- Default[train_id, ]
test  <- Default[-train_id, ]

# Procjena na trening skupu
mod_train <- glm(default ~ balance + student, data = train,
                 family = binomial)

# Predikcija na testnom skupu
vjer_test <- predict(mod_train, newdata = test, type = "response")
pred_test <- ifelse(vjer_test > 0.5, "Yes", "No")

# Matrica zabune
tab <- table(Predikcija = pred_test, Stvarno = test$default)
tab

# Mjere kvalitete
TP <- tab["Yes", "Yes"]
TN <- tab["No", "No"]
FP <- tab["Yes", "No"]
FN <- tab["No", "Yes"]

accuracy    <- (TP + TN) / sum(tab)
sensitivity <- TP / (TP + FN)    # osjetljivost / recall
specificity <- TN / (TN + FP)    # specifičnost

cat("Točnost:      ", round(accuracy, 3), "\n")
cat("Osjetljivost: ", round(sensitivity, 3), "\n")
cat("Specifičnost: ", round(specificity, 3), "\n")
```

### Promjena praga odlučivanja

```r
# Usporedba pragova 0.2, 0.5, 0.8 (iz zadataka)
pragovi <- c(0.2, 0.5, 0.8)

for (p in pragovi) {
  pred <- ifelse(vjer_test > p, "Yes", "No")
  tab  <- table(Predikcija = pred, Stvarno = test$default)
  TP   <- tab["Yes", "Yes"]
  TN   <- tab["No", "No"]
  FP   <- tab["Yes", "No"]
  FN   <- tab["No", "Yes"]
  cat("Prag =", p,
      "| Točnost:", round((TP+TN)/sum(tab), 3),
      "| Osjetljivost:", round(TP/(TP+FN), 3),
      "| Specifičnost:", round(TN/(TN+FP), 3), "\n")
}
# Manji prag → veća osjetljivost, manja specifičnost
# Veći prag → manja osjetljivost, veća specifičnost
```

### Simulirani podaci (iz zadataka)

```r
# Simulacija logističke regresije (iz zadataka Brborović)
set.seed(42)
n <- 200
x_sim <- runif(n, 0, 8)
# Pravi model: p(x) = e^(-3+0.8x) / (1 + e^(-3+0.8x))
p_pravi <- exp(-3 + 0.8*x_sim) / (1 + exp(-3 + 0.8*x_sim))
y_sim <- rbinom(n, size = 1, prob = p_pravi)

# Prikaz podataka
plot(x_sim, y_sim, pch = 16, col = "gray40",
     xlab = "x", ylab = "Y (0/1)")

# Procjena logističke regresije
mod_sim <- glm(y_sim ~ x_sim, family = binomial)
summary(mod_sim)

# Crtanje procijenjene krivulje
xseq <- seq(0, 8, length.out = 300)
pr_proc <- predict(mod_sim, newdata = data.frame(x_sim = xseq),
                   type = "response")
lines(xseq, pr_proc, col = "blue3", lwd = 2)

# Pravi model (za usporedbu)
p_pravi_krivulja <- exp(-3 + 0.8*xseq) / (1 + exp(-3 + 0.8*xseq))
lines(xseq, p_pravi_krivulja, col = "red3", lwd = 2, lty = 2)
legend("topleft", legend = c("Procijenjena", "Prava"),
       col = c("blue3", "red3"), lwd = 2)
```

## Interpretacija rezultata

**Koeficijent:**
> "Koeficijent uz `balance` iznosi 0.00550 i statistički je značajan (p < 0.001). To znači da povećanje balance za 1 dolara povećava log-izglede defaulta za 0.00550. Odds ratio = exp(0.00550) ≈ 1.0055, tj. svaki dolar više na kartici množi izglede defaulta s 1.0055. Porast od 100 dolara množi izglede s exp(0.55) ≈ 1.73."

**Devijanca:**
> "Null deviance iznosi 2920.6, a residual deviance 1596.5. Veliko smanjenje devijance uključivanjem varijable `balance` upućuje da ta varijabla bitno poboljšava model."

**Prag odlučivanja:**
> "Smanjenjem praga odlučivanja s 0.5 na 0.2 povećavamo osjetljivost modela (identificiramo više stvarnih pozitivnih slučajeva), ali istovremeno povećavamo broj lažno pozitivnih klasifikacija i smanjujemo specifičnost. U medicinskim primjenama gdje je važno ne propustiti bolest, koristili bismo niži prag."

## Najčešće greške

- **Ne navedéš `type = "response"`** u `predict()` → dobiješ logite (ne vjerojatnosti!)
- **Zaboraviš `family = binomial`** u `glm()` → R procjenjuje linearnu regresiju
- **Pokušavaš interpretirati koeficijent kao u linearnoj regresiji** – mora ići preko odds ratio
- **Visoka točnost kod nebalansiranih klasa** – trivijalni klasifikator može imati 97% točnosti ako je 97% negativnih

## Mini podsjetnik – Logistička regresija

| Što trebam | R kod |
|---|---|
| Izgraditi model | `glm(Y ~ X, family = binomial, data = df)` |
| Odds ratios | `exp(coef(model))` |
| Procijeniti vjerojatnosti | `predict(model, type = "response")` |
| Klasificirati | `ifelse(vjer > 0.5, "Yes", "No")` |
| Matrica zabune | `table(predikcija, stvarno)` |
| Usporediti modele | `AIC(m1, m2)` |

---

# 7. Mjere kvalitete klasifikacije

## Teorija

**Matrica zabune (confusion matrix)** za binarnu klasifikaciju:

```
                Stvarna NO    Stvarna YES
Predviđena NO      TN            FN
Predviđena YES     FP            TP
```

- **TP** (True Positive) – ispravno klasificirani pozitivni
- **TN** (True Negative) – ispravno klasificirani negativni
- **FP** (False Positive) – negativni pogrešno klasificirani kao pozitivni (lažni alarm)
- **FN** (False Negative) – pozitivni pogrešno klasificirani kao negativni (propušteni slučajevi)

**Formule:**

| Mjera | Formula | Kada je važna |
|---|---|---|
| **Točnost (Accuracy)** | (TP+TN)/(TP+TN+FP+FN) | Balansirane klase |
| **Osjetljivost (Sensitivity/Recall)** | TP/(TP+FN) | Kada je skupo propustiti pozitivan |
| **Specifičnost (Specificity)** | TN/(TN+FP) | Kada je skupo lažni alarm |
| **Preciznost (Precision)** | TP/(TP+FP) | Kada je skupo lažni alarm |
| **F1 score** | 2·(Prec·Recall)/(Prec+Recall) | Nebalansirane klase |

## R kodovi

```r
# Kompletna funkcija za izračun svih mjera
izracunaj_mjere <- function(stvarno, predikcija, pozitivna_klasa = "Yes") {
  tab <- table(Predikcija = predikcija, Stvarno = stvarno)
  
  # Siguran pristup tablici
  TP <- ifelse(!is.na(tab[pozitivna_klasa, pozitivna_klasa]),
               tab[pozitivna_klasa, pozitivna_klasa], 0)
  negativna_klasa <- setdiff(rownames(tab), pozitivna_klasa)
  TN <- ifelse(length(negativna_klasa) > 0,
               tab[negativna_klasa, negativna_klasa], 0)
  FP <- ifelse(!is.na(tab[pozitivna_klasa, negativna_klasa]),
               tab[pozitivna_klasa, negativna_klasa], 0)
  FN <- ifelse(!is.na(tab[negativna_klasa, pozitivna_klasa]),
               tab[negativna_klasa, pozitivna_klasa], 0)
  
  acc  <- (TP + TN) / sum(tab)
  sens <- TP / (TP + FN)
  spec <- TN / (TN + FP)
  prec <- TP / (TP + FP)
  
  cat("Matrica zabune:\n"); print(tab)
  cat("\nTočnost:      ", round(acc, 4))
  cat("\nOsjetljivost: ", round(sens, 4))
  cat("\nSpecifičnost: ", round(spec, 4))
  cat("\nPreciznost:   ", round(prec, 4), "\n")
}

# Primjer korištenja
izracunaj_mjere(test$default, pred_test, pozitivna_klasa = "Yes")

# Ručni izračun (brži na ispitu)
tab <- table(Predikcija = pred_test, Stvarno = test$default)
TP  <- tab["Yes", "Yes"]
TN  <- tab["No",  "No"]
FP  <- tab["Yes", "No"]
FN  <- tab["No",  "Yes"]

accuracy    <- (TP + TN) / sum(tab)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
```

### ROC krivulja i AUC

```r
# install.packages("pROC")
library(pROC)

# Procijenjene vjerojatnosti (trebaju biti za pozitivnu klasu)
roc_obj <- roc(test$default, vjer_test)
plot(roc_obj, col = "blue3", lwd = 2,
     main = "ROC krivulja")
auc(roc_obj)  # Vrijednost između 0.5 (slučajno) i 1 (savršeno)

# AUC interpretacija:
# 0.5 = model ne razlikuje klase bolje od slučajnog
# 0.7–0.8 = prihvatljiv
# 0.8–0.9 = dobar
# > 0.9 = odličan
```

## Interpretacija

> "Model postiže točnost od 97.3%. Međutim, osjetljivost iznosi svega 33.4%, što znači da model prepoznaje samo trećinu stvarnih slučajeva defaulta. To je problema jer je klasa `Yes` rijetka (3.3%) – trivijalni klasifikator koji sve klasificira kao `No` postigao bi točnost od 96.7%."

> "AUC = 0.947 upućuje da model ima izvrsnu sposobnost razlikovanja između slučajeva defaulta i ne-defaulta, neovisno o odabranom pragu odlučivanja."

**Rijetka klasa:**
> "Kada je klasa `Yes` zastupljena s manje od 5% opažanja, točnost klasifikacije je zavaravajuća mjera kvalitete. Preporučuje se pratiti osjetljivost i specifičnost odvojeno, te koristiti AUC kao sveobuhvatnu mjeru."

---

# 8. KNN klasifikacija

## Teorija

KNN (K-Nearest Neighbours) je **neparametarski** klasifikator koji ne pretpostavlja nikakav oblik modela. Za novo opažanje pronalazi K najbližih susjeda u trening skupu i dodjeljuje klasu većine.

- **Malo K (npr. K=1)**: fleksibilna granica odlučivanja, visoka varijanca, prenaučenost
- **Veliko K (npr. K=25)**: glatka granica, veća pristranost, generalizira bolje

**VAŽNO:** Prediktore treba standardizirati prije KNN-a!

## R kodovi

```r
library(class)     # za knn()
library(ISLR2)
data(OJ)

# Priprema podataka – odabir prediktora
OJ2 <- OJ[, c("Purchase", "LoyalCH", "PriceDiff")]

# Trening/testni skup
set.seed(123)
train_id <- sample(1:nrow(OJ2), 800)
OJ_train <- OJ2[train_id, ]
OJ_test  <- OJ2[-train_id, ]

# Standardizacija (OBAVEZNO za KNN!)
# Koristimo SAMO parametre trening skupa za standardizaciju
centar <- apply(OJ_train[, c("LoyalCH", "PriceDiff")], 2, mean)
skala  <- apply(OJ_train[, c("LoyalCH", "PriceDiff")], 2, sd)

X_train <- scale(OJ_train[, c("LoyalCH", "PriceDiff")],
                 center = centar, scale = skala)
X_test  <- scale(OJ_test[,  c("LoyalCH", "PriceDiff")],
                 center = centar, scale = skala)

Y_train <- OJ_train$Purchase
Y_test  <- OJ_test$Purchase

# KNN za različite vrijednosti k (iz materijala Brborović)
knn_1  <- knn(train = X_train, test = X_test, cl = Y_train, k = 1)
knn_5  <- knn(train = X_train, test = X_test, cl = Y_train, k = 5)
knn_15 <- knn(train = X_train, test = X_test, cl = Y_train, k = 15)

# Točnost
cat("k=1  točnost:", round(mean(knn_1  == Y_test), 3), "\n")
cat("k=5  točnost:", round(mean(knn_5  == Y_test), 3), "\n")
cat("k=15 točnost:", round(mean(knn_15 == Y_test), 3), "\n")

# Matrica zabune za k=5
tab_knn5 <- table(Predikcija = knn_5, Klasa = Y_test)
tab_knn5

# Osjetljivost i specifičnost za k=5 (Purchase = CH vs MM)
TP <- tab_knn5["MM", "MM"]
TN <- tab_knn5["CH", "CH"]
FP <- tab_knn5["MM", "CH"]
FN <- tab_knn5["CH", "MM"]
cat("Osjetljivost:", round(TP/(TP+FN), 3), "\n")
cat("Specifičnost:", round(TN/(TN+FP), 3), "\n")
```

## Interpretacija

> "KNN klasifikator s k=5 postiže točnost od 82.2% na testnom skupu, što je usporedivo s logističkom regresijom (81.9%). Za k=1 postižemo bolji rezultat na trening skupu (100%), ali lošiji na testnom, što ukazuje na prenaučenost. Povećanjem k na 15 granica odlučivanja postaje glađa i model generalijzira bolje."

---

# 9. Poissonova regresija

## Teorija

Koristimo Poissonovu regresiju kada je odzivna varijabla **brojevna** (count data: 0, 1, 2, ...). Linearna regresija nije prikladna jer može davati negativne vrijednosti.

Model: `Y | X ~ Poisson(λ(x))`

Jednadžba: `log(λ(x)) = β₀ + β₁X₁ + ... + βₚXₚ`

Pa je: `λ(x) = exp(β₀ + β₁X₁ + ... + βₚXₚ)` — uvijek pozitivno!

**Karakteristika Poissonove razdiobe:** E(Y) = Var(Y) = λ

**Prekomjerna disperzija** (overdispersion): Var(Y) >> E(Y) → Poisson model je previše restriktivan → koristiti quasipoisson.

## Ključni pojmovi

| Pojam | Značenje |
|---|---|
| **λ** | Očekivani broj događaja |
| **log link** | log(λ) = linearni prediktor |
| **exp(βᵢ)** | Faktor kojim se množi λ pri povećanju Xᵢ za 1 |
| **Prekomjerna disperzija** | Var(Y) > E(Y); omjer devijance/df >> 1 |
| **Devijanca/df** | Omjer > 1 → potencijalna prekomjerna disperzija; > 2–3 → ozbiljan problem |
| **quasipoisson** | Korigira SE za prekomjernu disperziju, ali zadržava iste koeficijente |

## R kodovi

### Simulirani primjer (iz materijala)

```r
# Simulacija (iz materijala Brborović)
set.seed(123)
n <- 200
x <- runif(n, 0, 5)
lambda <- exp(1 + 0.3*x)   # log(λ) = 1 + 0.3x
y <- rpois(n, lambda = lambda)
pod_pois <- data.frame(x = x, y = y)

# Vizualizacija
plot(pod_pois$x, pod_pois$y, pch = 16, col = "gray40",
     xlab = "x", ylab = "Broj događaja")

# Poissonova regresija
model_pois1 <- glm(y ~ x, family = poisson, data = pod_pois)
summary(model_pois1)

# Interpretacija koeficijenta
exp(coef(model_pois1)["x"])
# Povećanje x za 1 množi λ s exp(β₁)

# Provjera prekomjerne disperzije
deviance(model_pois1) / df.residual(model_pois1)
# ≈ 1 je OK; >> 1 je problem
```

### Bikeshare primjer (iz materijala)

```r
library(ISLR2)
data("Bikeshare")

# Boxplotovi (vizualna istraga)
par(mfrow = c(2, 2))
boxplot(bikers ~ mnth, data = Bikeshare, col = "lightskyblue2",
        main = "Korištenje bicikala po mjesecima")
boxplot(bikers ~ hr, data = Bikeshare, col = "lightskyblue2",
        main = "Korištenje bicikala po satima")

# Linearni model (za usporedbu)
model_lm <- lm(bikers ~ mnth + hr + workingday + temp + weathersit,
               data = Bikeshare)
range(predict(model_lm))  # Može biti negativno! Problem.

# Poissonova regresija
model_pois <- glm(bikers ~ mnth + hr + workingday + temp + weathersit,
                  data = Bikeshare, family = poisson)
summary(model_pois)

# Sve procijenjene vrijednosti su pozitivne
range(predict(model_pois, type = "response"))

# Interpretacija temp koeficijenta
exp(coef(model_pois)["temp"])
# Porast temp za 1 jedninicu (normaliziranoj skali) množi λ s ...

# Interpretacija za promjenu 0.1 (preglednije za normaliziranu varijablu)
exp(0.1 * coef(model_pois)["temp"])
# ~ 1.082 → porast za 0.1 povećava korištenje bicikala za 8.2%

# Faktorske varijable – omjeri vs referentna kategorija
exp(coef(model_pois)[grep("^weathersit", names(coef(model_pois)))])
# clear je referentna kategorija
```

### Provjera prekomjerne disperzije i quasipoisson

```r
# Provjera prekomjerne disperzije
deviance(model_pois) / df.residual(model_pois)
# Ako >> 1 (npr. 26.5 za Bikeshare) → prekomjerna disperzija!

# Rješenje 1: quasipoisson (isti koeficijenti, veće SE)
model_quasi <- glm(bikers ~ mnth + hr + workingday + temp + weathersit,
                   data = Bikeshare, family = quasipoisson)
summary(model_quasi)

# Usporedba SE (quasipoisson vs poisson)
# quasipoisson ima veće standardne pogreške
# To je konzervativnija i ispravnija procjena

# Simulirani primjer prekomjerne disperzije (iz zadataka)
set.seed(42)
n <- 200
x_pod <- runif(n, 0, 5)
U <- rnorm(n, 0, sd = 0.5)  # slučajni efekt
log_lambda <- 1 + 0.3*x_pod + U
y_over <- rpois(n, lambda = exp(log_lambda))

mod_over <- glm(y_over ~ x_pod, family = poisson)
deviance(mod_over) / df.residual(mod_over)  # > 1 → prekomjerna disperzija

mod_quasi_over <- glm(y_over ~ x_pod, family = quasipoisson)
# Usporedi summary-je
summary(mod_over)$coefficients
summary(mod_quasi_over)$coefficients  # Veće SE!
```

### Warpbreaks primjer (iz zadataka)

```r
data(warpbreaks)
str(warpbreaks)
summary(warpbreaks)

# Poissonova regresija s faktorima
mod_wb <- glm(breaks ~ wool + tension, data = warpbreaks,
              family = poisson)
summary(mod_wb)

# Omjeri – eksponentirani koeficijenti
exp(coef(mod_wb))
# exp(woolB) = 0.84 → vrsta vune B ima 84% lomova od referentne vrste A

# Provjera prekomjerne disperzije
deviance(mod_wb) / df.residual(mod_wb)

# Usporedba Poisson vs quasipoisson
mod_wb_quasi <- glm(breaks ~ wool + tension, data = warpbreaks,
                    family = quasipoisson)
summary(mod_wb_quasi)$coefficients  # Veće SE
```

## Interpretacija rezultata

**Koeficijent:**
> "Koeficijent uz `temp` iznosi 0.785. Eksponenciranjem dobivamo e^0.785 ≈ 2.19. To znači da se očekivani broj korisnika bicikala multiplicira s 2.19 pri porastu normalizirane temperature za 1 jedinicu, uz ostale prediktore fiksirane."

**Prekomjerna disperzija:**
> "Omjer devijance i stupnjeva slobode iznosi 26.5, što je znatno veće od 1 i upućuje na ozbiljnu prekomjernu disperziju. U osnovnom Poissonovom modelu standardne pogreške koeficijenata su podcijenjene. Preporučuje se primjena quasipoisson modela."

**Quasipoisson:**
> "U quasipoisson modelu koeficijenti ostaju jednaki kao u Poissonovom modelu, ali standardne pogreške su veće za faktor √φ (gdje je φ parametar disperzije). To rezultira konzervativnijim zaključcima o statističkoj značajnosti prediktora."

## Mini podsjetnik – Poissonova regresija

| Što trebam | R kod |
|---|---|
| Model | `glm(Y ~ X, family = poisson, data = df)` |
| Interpretiraj koeficijente | `exp(coef(model))` |
| Provjeri prekomjernu disperziju | `deviance(m)/df.residual(m)` |
| Korekcija za disperziju | `glm(..., family = quasipoisson)` |
| Predikcija | `predict(model, type = "response")` |

---

# 10. Cross-validacija

## Teorija

**Cross-validacija (CV)** procjenjuje testnu grešku modela bez potrebe za zasebnim testnim skupom. Ključna ideja: koristimo dio podataka za učenje, a drugi dio za provjeru, i mijenjamo koji dio je koji.

### Tri pristupa (iz materijala Brborović):

**1. Validation set approach** – najjednostavniji, visoka varijabilnost rezultata

**2. LOOCV (Leave-One-Out CV)** – izostavimo jedno opažanje, model procjenimo na n-1, ponavljamo za svaki. Stabilno, ali skupo.

**3. k-fold CV** – dijelimo na k dijelova, svaki po jednom bude testni. k=5 ili k=10 je standard.

**LOOCV formula:** `CV(n) = (1/n) Σ (yᵢ - ŷ(i))²`

**k-fold formula:** `CV(k) = (1/k) Σ (greška na j-tom validacijskom skupu)`

## R kodovi

### Validation set approach (iz materijala)

```r
library(ISLR2)
data(Auto)
set.seed(123)
n <- nrow(Auto)
train_id <- sample(1:n, size = n/2)
train <- Auto[train_id, ]
test  <- Auto[-train_id, ]

mod1 <- lm(mpg ~ horsepower, data = train)
mod2 <- lm(mpg ~ horsepower + I(horsepower^2), data = train)
mod3 <- lm(mpg ~ horsepower + I(horsepower^2) + I(horsepower^3), data = train)

mse1 <- mean((test$mpg - predict(mod1, newdata = test))^2)
mse2 <- mean((test$mpg - predict(mod2, newdata = test))^2)
mse3 <- mean((test$mpg - predict(mod3, newdata = test))^2)

cat("Linearni MSE:", round(mse1, 2), "\n")
cat("Kvadratni MSE:", round(mse2, 2), "\n")
cat("Kubični MSE:", round(mse3, 2), "\n")
```

### LOOCV i k-fold CV (iz materijala)

```r
library(boot)

# Modele moramo definirati s glm() (ne lm()) za cv.glm()
glm1 <- glm(mpg ~ horsepower, data = Auto)
glm2 <- glm(mpg ~ horsepower + I(horsepower^2), data = Auto)
glm3 <- glm(mpg ~ horsepower + I(horsepower^2) + I(horsepower^3), data = Auto)

# LOOCV
cv_loocv1 <- cv.glm(Auto, glm1)$delta[1]
cv_loocv2 <- cv.glm(Auto, glm2)$delta[1]
cv_loocv3 <- cv.glm(Auto, glm3)$delta[1]
cat("LOOCV MSE (lin, kv, kub):", cv_loocv1, cv_loocv2, cv_loocv3, "\n")

# 10-fold CV
set.seed(123)
cv_kfold1 <- cv.glm(Auto, glm1, K = 10)$delta[1]
cv_kfold2 <- cv.glm(Auto, glm2, K = 10)$delta[1]
cv_kfold3 <- cv.glm(Auto, glm3, K = 10)$delta[1]
cat("10-fold CV MSE:", cv_kfold1, cv_kfold2, cv_kfold3, "\n")

# CV za odabir stupnja polinoma (1-6) – iz materijala
set.seed(123)
cv_error <- numeric(6)
for (d in 1:6) {
  model <- glm(mpg ~ poly(horsepower, d), data = Auto)
  cv_error[d] <- cv.glm(Auto, model, K = 10)$delta[1]
}
cv_error

# Prikaz
plot(1:6, cv_error, type = "b", pch = 16,
     xlab = "Stupanj polinoma", ylab = "10-fold CV MSE",
     main = "Odabir stupnja polinoma pomoću CV")
which.min(cv_error)  # Optimalni stupanj polinoma
```

## Interpretacija

> "10-fold cross-validacija pokazuje da kvadratni model (cv MSE = 18.56) ima manju testnu grešku od linearnog (cv MSE = 24.23) i kubičnog (cv MSE = 18.83). CV krivulja pokazuje minimum na stupnju 2, što upućuje da kvadratni polinom najprikladnije opisuje odnos između `mpg` i `horsepower` bez prenaučenosti."

> "Validation set approach daje varijabilne rezultate koji ovise o slučajnoj podjeli podataka. 10-fold CV je stabilnija procjena testne greške i preporučuje se u praksi."

**Cross-validacija vs Bootstrap:**
> "Cross-validacija se koristi za procjenu predikcijske greške i odabir modela. Bootstrap se koristi za procjenu nesigurnosti procjenitelja (standardne greške i intervali pouzdanosti)."

## Mini podsjetnik – CV

| Metoda | Kad koristiti | R kod |
|---|---|---|
| Validation set | Brza provjera | `lm(... train)`; `predict(... test)` |
| LOOCV | Malo podataka | `cv.glm(data, model)$delta[1]` |
| 10-fold CV | Standard | `cv.glm(data, model, K=10)$delta[1]` |

---

# 11. Bootstrap

## Teorija

**Bootstrap** procjenjuje varijabilnost statističkih procjenitelja uzorkovanjem s ponavljanjem iz dostupnih podataka. Ne zahtijeva teorijske pretpostavke o raspodjeli.

Ideja: dostupni uzorak ≈ populacija → uzorkujemo s ponavljanjem B puta → dobivamo B procjena → standardna devijacija tih procjena = bootstrap SE.

**Bootstrap SE formula:** `SE_boot(θ̂) = sd(θ̂*₁, ..., θ̂*ᵦ)`

**Bootstrap interval pouzdanosti (95%):** `[q₀.₀₂₅, q₀.₉₇₅]` bootstrap distribucije

## R kodovi

### Ručni bootstrap (iz materijala)

```r
# Bootstrap za aritmetičku sredinu (iz materijala)
set.seed(123)
x_norm <- rnorm(30, mean = 10, sd = 2)

# Teorijska SE
se_teor <- sd(x_norm) / sqrt(length(x_norm))
cat("Teorijska SE:", round(se_teor, 4), "\n")

# Bootstrap SE
B <- 1000
boot_sredina <- numeric(B)
for (b in 1:B) {
  x_zv <- sample(x_norm, size = length(x_norm), replace = TRUE)
  boot_sredina[b] <- mean(x_zv)
}
cat("Bootstrap SE:", round(sd(boot_sredina), 4), "\n")

# Histogram bootstrap distribucije
hist(boot_sredina, breaks = 30, col = "lightskyblue2",
     main = "Bootstrap distribucija sredine",
     xlab = "Bootstrap sredine")

# Bootstrap IP (interval pouzdanosti)
quantile(boot_sredina, c(0.025, 0.975))
```

### Bootstrap s paketom `boot` (iz materijala)

```r
library(boot)
set.seed(123)

# Za sredinu
mean_fun <- function(data, index) mean(data[index])
boot_mean <- boot(data = x_norm, statistic = mean_fun, R = 1000)
boot_mean
plot(boot_mean)
boot.ci(boot_mean, type = c("perc", "basic"))

# Za medijan
x_exp <- rexp(50, rate = 1)
median_fun <- function(data, index) median(data[index])
boot_med <- boot(data = x_exp, statistic = median_fun, R = 2000)
boot_med
boot.ci(boot_med, type = c("perc", "basic"))
```

### Bootstrap u regresiji (iz materijala)

```r
library(ISLR2)
data(Auto)
library(boot)

# Bootstrap za koeficijente linearne regresije
boot_lm_fun <- function(data, index) {
  data_boot <- data[index, ]
  model <- lm(mpg ~ horsepower, data = data_boot)
  coef(model)
}

set.seed(123)
boot_lm <- boot(Auto, boot_lm_fun, R = 1000)
boot_lm

# Bootstrap SE
apply(boot_lm$t, 2, sd)

# Usporedba s klasičnim SE iz lm()
summary(lm(mpg ~ horsepower, data = Auto))$coefficients

# Bootstrap IP za koeficijent uz horsepower (index=2)
boot.ci(boot_lm, type = c("perc", "basic"), index = 2)
```

## Interpretacija

> "Bootstrap standardna greška koeficijenta uz `horsepower` iznosi 0.0064, što je blisko teorijskoj vrijednosti 0.0064 iz `summary()`. U ovom slučaju su pretpostavke linearne regresije zadovoljene pa su obje procjene sukladne."

> "95% bootstrap interval pouzdanosti za medijan uzorka iznosi [0.52, 0.89]. Budući da je medijan procjenitelj za koji nemamo jednostavnu teorijsku formulu, bootstrap je posebno koristan pristup."

---

# 12. Stabla odluke

## Teorija

Stabla odluke dijele prediktorski prostor na regije rekurzivnim binarnim podjelama. Svako opažanje dobiva predikciju = prosječna vrijednost (regresija) ili modus (klasifikacija) u svojoj regiji.

**Prednosti:** interpretabilnost, vizualizacija, radi s mješovitim tipovima podataka, ne zahtijeva standardizaciju.

**Nedostaci:** nestabilnost (visoka varijanca), nagnuće ka prenaučenosti.

**Rezanje stabla (pruning):** smanjuje prenaučenost; koristimo CV za odabir složenosti.

**Za klasifikaciju koristi se:**
- Gini indeks: `G = Σ pₖ(1 - pₖ)` (manji = čistiji čvor)
- Entropija: `-Σ pₖ log(pₖ)`

## R kodovi

```r
# install.packages("tree")
library(tree)
library(ISLR2)

# --- Klasifikacijsko stablo ---
data(OJ)

set.seed(123)
train_id <- sample(1:nrow(OJ), 800)
OJ_train <- OJ[train_id, ]
OJ_test  <- OJ[-train_id, ]

# Izgradnja stabla
stablo <- tree(Purchase ~ ., data = OJ_train)
summary(stablo)
# Terminal nodes: broj listova
# Misclassification error rate: trening greška

# Vizualizacija stabla
plot(stablo)
text(stablo, pretty = 0, cex = 0.8)

# Predikcija na testnom skupu
pred_stablo <- predict(stablo, newdata = OJ_test, type = "class")
tab_stablo  <- table(Predikcija = pred_stablo, Stvarno = OJ_test$Purchase)
tab_stablo
mean(pred_stablo == OJ_test$Purchase)  # točnost

# Rezanje stabla (pruning) pomoću CV
cv_stablo <- cv.tree(stablo, FUN = prune.misclass)
cv_stablo
# Grafički prikaz
plot(cv_stablo$size, cv_stablo$dev, type = "b",
     xlab = "Veličina stabla", ylab = "CV pogreška",
     main = "CV za odabir veličine stabla")

# Reži stablo na optimalnu veličinu
optimalna_velicina <- cv_stablo$size[which.min(cv_stablo$dev)]
stablo_rez <- prune.misclass(stablo, best = optimalna_velicina)
plot(stablo_rez)
text(stablo_rez, pretty = 0, cex = 0.8)

# Evaluacija rezanog stabla
pred_rez <- predict(stablo_rez, newdata = OJ_test, type = "class")
mean(pred_rez == OJ_test$Purchase)

# --- Regresijsko stablo ---
data(Boston)
set.seed(123)
n <- nrow(Boston)
train_id <- sample(1:n, n/2)
Boston_train <- Boston[train_id, ]
Boston_test  <- Boston[-train_id, ]

stablo_reg <- tree(medv ~ ., data = Boston_train)
plot(stablo_reg); text(stablo_reg, pretty = 0)

pred_reg <- predict(stablo_reg, newdata = Boston_test)
mse_stablo <- mean((pred_reg - Boston_test$medv)^2)
cat("Testni MSE stabla:", round(mse_stablo, 2), "\n")
```

## Interpretacija

> "Klasifikacijsko stablo s 8 listova postiže trening grešku od 16.5% i testnu grešku od 19.2%. Nakon rezanja na 5 listova (određeno 10-fold CV), testna greška iznosi 18.9%, uz znatno jednostavnije stablo koje je lakše interpretirati."

> "Varijabla `LoyalCH` se pojavljuje kao najvažnija u prvoj grani stabla, što sugerira da je lojalnost marki najvažniji prediktor odluke o kupnji."

---

# 13. Random Forest

## Teorija

**Random Forest** je ansambl metoda – gradi mnogo stabala odluke i kombinira njihove predikcije (glasanje za klasifikaciju, prosjek za regresiju).

**Ključna ideja:** svako stablo se gradi na bootstrap uzorku podataka, a na svakom čvoru se razmatra samo **random podskup od m prediktora** (ne svi). To smanjuje korelaciju između stabala i smanjuje varijancu.

- **m = √p** za klasifikaciju (p = ukupan broj prediktora)
- **m = p/3** za regresiju

**OOB (Out-of-Bag)** greška: automatska procjena testne greške bez potrebe za zasebnim testnim skupom.

**Važnost varijabli (variable importance):** mjeri doprinos svakog prediktora smanjenju greške.

## R kodovi

```r
# install.packages("randomForest")
library(randomForest)
library(ISLR2)

# --- Klasifikacijski Random Forest ---
data(OJ)
set.seed(123)
train_id <- sample(1:nrow(OJ), 800)
OJ_train <- OJ[train_id, ]
OJ_test  <- OJ[-train_id, ]

# mtry = broj prediktora po čvoru (sqrt(p) za klasifikaciju)
p <- ncol(OJ_train) - 1
rf_oj <- randomForest(Purchase ~ .,
                       data = OJ_train,
                       mtry = floor(sqrt(p)),
                       ntree = 500,
                       importance = TRUE)
rf_oj
# OOB estimate of error rate: automatska procjena greške

# Predikcija na testnom skupu
pred_rf <- predict(rf_oj, newdata = OJ_test)
tab_rf  <- table(Predikcija = pred_rf, Stvarno = OJ_test$Purchase)
tab_rf
mean(pred_rf == OJ_test$Purchase)  # točnost

# Važnost varijabli
importance(rf_oj)
varImpPlot(rf_oj, main = "Važnost prediktora - Random Forest")
# MeanDecreaseAccuracy: koliko se točnost smanji bez te varijable
# MeanDecreaseGini: smanjenje nečistoće (Gini indeks)

# --- Regresijski Random Forest ---
data(Boston)
set.seed(123)
n <- nrow(Boston)
train_id <- sample(1:n, n/2)
Boston_train <- Boston[train_id, ]
Boston_test  <- Boston[-train_id, ]

p_bos <- ncol(Boston_train) - 1  # broj prediktora

# mtry = p/3 za regresiju (ili probati više vrijednosti)
rf_bos <- randomForest(medv ~ .,
                        data = Boston_train,
                        mtry = floor(p_bos/3),
                        ntree = 500,
                        importance = TRUE)
rf_bos

pred_rf_bos <- predict(rf_bos, newdata = Boston_test)
mse_rf <- mean((pred_rf_bos - Boston_test$medv)^2)
cat("Random Forest testni MSE:", round(mse_rf, 2), "\n")

varImpPlot(rf_bos, main = "Važnost prediktora - Boston")

# Optimizacija mtry
mse_mtry <- numeric(p_bos)
for (m in 1:p_bos) {
  set.seed(123)
  rf_temp <- randomForest(medv ~ ., data = Boston_train, mtry = m, ntree = 300)
  pred_temp <- predict(rf_temp, newdata = Boston_test)
  mse_mtry[m] <- mean((pred_temp - Boston_test$medv)^2)
}
plot(1:p_bos, mse_mtry, type = "b", pch = 16,
     xlab = "mtry", ylab = "Testni MSE",
     main = "Odabir mtry parametra")
which.min(mse_mtry)  # Optimalni mtry
```

## Interpretacija

> "Random Forest s mtry=3 i 500 stabala postiže testni MSE od 11.58, što je znatno manje od jednog stabla odluke (MSE = 26.4). OOB greška iznosi 13.7%, što je konzistentno s testnom greškom. Varijabla `lstat` (% niskog statusa stanovništva) i `rm` (prosječan broj soba) identificiraju se kao najvažniji prediktori prema MeanDecreaseGini kriteriju."

**Prenaučenost:**
> "Prednost Random Foresta je što agregiranjem velikog broja stabala smanjuje varijancu i problem prenaučenosti koji je karakterističan za pojedinačna stabla. Bagging (mtry = p) i Random Forest (mtry < p) smanjuju korelaciju između stabala, što dodatno poboljšava performanse."

---

# 14. Support Vector Machine (SVM)

## Teorija

**SVM** traži **optimalnu granicu odlučivanja** (hyperplane) koja maksimizira **margu** (razmak između granice i najbližih opažanja iz svake klase). Ta najbliža opažanja zovu se **potporni vektori** (support vectors).

**Ključne verzije:**
- **Hard margin SVM** – savršena separacija (rijetko u praksi)
- **Soft margin SVM (C-SVM)** – dozvoljava neke pogreške; **C = cost** parametar
  - Malo C → šira marga, više pogrešaka na trening skupu, manje prenaučenosti
  - Veliko C → uska marga, manje tolerancije, više prenaučenosti
- **SVM s kernel trikom** – za nelinearne granice
  - Linearni kernel: `kernel = "linear"`
  - Radijalni (RBF) kernel: `kernel = "radial"` (parametar `gamma`)
  - Polinomijalni kernel: `kernel = "polynomial"` (parametar `degree`)

## R kodovi

```r
# install.packages("e1071")
library(e1071)
library(ISLR2)

# --- Klasifikacijski SVM ---
data(OJ)
set.seed(123)
train_id <- sample(1:nrow(OJ), 800)
OJ_train <- OJ[train_id, ]
OJ_test  <- OJ[-train_id, ]

# Linearni kernel
svm_lin <- svm(Purchase ~ LoyalCH + PriceDiff,
               data = OJ_train,
               kernel = "linear",
               cost = 1,
               scale = TRUE)
summary(svm_lin)
# Number of Support Vectors: opažanja koja definiraju granicu

# Predikcija i evaluacija
pred_svm <- predict(svm_lin, newdata = OJ_test)
tab_svm  <- table(Predikcija = pred_svm, Stvarno = OJ_test$Purchase)
tab_svm
mean(pred_svm == OJ_test$Purchase)

# Radijalni kernel (za nelinearne granice)
svm_rad <- svm(Purchase ~ LoyalCH + PriceDiff,
               data = OJ_train,
               kernel = "radial",
               cost = 1,
               gamma = 1,
               scale = TRUE)
pred_svm_rad <- predict(svm_rad, newdata = OJ_test)
mean(pred_svm_rad == OJ_test$Purchase)

# Tuning (odabir optimalnih parametara C i gamma)
set.seed(123)
tune_out <- tune(svm,
                 Purchase ~ LoyalCH + PriceDiff,
                 data = OJ_train,
                 kernel = "radial",
                 ranges = list(
                   cost  = c(0.1, 1, 10, 100),
                   gamma = c(0.01, 0.1, 1)
                 ))
summary(tune_out)
tune_out$best.parameters  # Optimalni parametri
tune_out$best.performance # Minimalna CV greška

# Model s optimalnim parametrima
svm_opt <- tune_out$best.model
pred_opt <- predict(svm_opt, newdata = OJ_test)
mean(pred_opt == OJ_test$Purchase)

# --- Usporedba kernela na generičkim podacima ---
set.seed(42)
n <- 200
x1 <- rnorm(n); x2 <- rnorm(n)
y  <- factor(ifelse(x1^2 + x2^2 < 1.5, "A", "B"))
df_svm <- data.frame(x1 = x1, x2 = x2, y = y)

# Linearan kernel – loš za kružne podatke
svm_lin_test <- svm(y ~ x1 + x2, data = df_svm,
                    kernel = "linear", cost = 1)
# Radijalni kernel – bolji za kružne granice
svm_rad_test <- svm(y ~ x1 + x2, data = df_svm,
                    kernel = "radial", cost = 1, gamma = 1)
mean(predict(svm_lin_test) == df_svm$y)
mean(predict(svm_rad_test) == df_svm$y)
```

## Interpretacija

> "SVM s linearnim kernelom i C=1 postiže točnost od 83.1% na testnom skupu. Tuning radijalnog kernela daje optimalne parametre C=10, gamma=0.1 s CV pogreškom od 15.2%. Model s optimalnim parametrima postiže točnost od 85.0% na testnom skupu, što je bolje od linearnog kernela, što sugerira da granica odlučivanja nije potpuno linearna."

**Parametar C:**
> "Smanjivanjem parametra C (cost) s 100 na 0.1 povećava se marga i broj potpornih vektora, a model postaje manje sklon prenaučenosti. Tuning pokazuje da optimalno C ovisi o konkretnim podacima."

---

# 15. PCA – Analiza glavnih komponenti

## Teorija

**PCA** (Principal Component Analysis) je metoda **redukcije dimenzionalnosti**. Transformira originalne (možda visoko korelirane) varijable u novi, manji skup nekorelirani varijabli – **glavne komponente (PC)**.

- **PC1** – smjer najveće varijabilnosti
- **PC2** – smjer najveće varijabilnosti koji je okomit na PC1
- ...itd.

**Loadings (opterećenja)**: koliko svaka originalna varijabla doprinosi svakoj PC.

**Proportion of Variance Explained (PVE)**: udio varijabilnosti objašnjen svakom PC.

**Scree plot**: grafički prikaz PVE – tražimo "lakat" (elbow) gdje se PVE prestaje naglo smanjivati.

**VAŽNO:** Standardizirati podatke prije PCA ako varijable imaju različite skale!

## R kodovi

```r
# Primjer s USArrests datasetom (klasični primjer za PCA)
data(USArrests)
str(USArrests)
summary(USArrests)

# PCA – obavezno scale = TRUE ako su varijable na različitim skalama
pca_us <- prcomp(USArrests, scale = TRUE)

# Pregled rezultata
summary(pca_us)
# Importance of components:
# Standard deviation: SD svake PC
# Proportion of Variance: udio varijance po PC
# Cumulative Proportion: kumulativni udio

# Loadings (opterećenja, rotacijska matrica)
pca_us$rotation
# Svaka kolona je PC, svaki red je originalna varijabla
# Znak loadinga = smjer veze

# Scores (koordinate opažanja u novom prostoru)
head(pca_us$x)

# Vizualizacija – biplot
biplot(pca_us, scale = 0,
       main = "Biplot PCA – USArrests")
# Strelice = originalne varijable (loadings)
# Točke = opažanja (SAD-ovi)
# Strelice u istom smjeru = pozitivno korelirane varijable

# Scree plot (varijanca po komponenti)
var_exp <- pca_us$sdev^2                     # varijance
pve     <- var_exp / sum(var_exp)            # PVE
cum_pve <- cumsum(pve)                       # kumulativni PVE

par(mfrow = c(1, 2))
plot(pve, type = "b", pch = 16, col = "blue3",
     xlab = "Redni broj PC", ylab = "Udio varijance (PVE)",
     main = "Scree plot – PVE")
abline(h = 0.1, col = "red", lty = 2)

plot(cum_pve, type = "b", pch = 16, col = "darkgreen",
     xlab = "Redni broj PC", ylab = "Kumulativni PVE",
     main = "Kumulativni PVE")
abline(h = 0.8, col = "red", lty = 2)  # linija na 80%
```

### Primjer s College datasetom

```r
library(ISLR2)
data(College)

# Uzimamo numeričke varijable
college_num <- College[, sapply(College, is.numeric)]

# PCA (scale = TRUE jer su varijable na jako različitim skalama)
pca_college <- prcomp(college_num, scale = TRUE)

summary(pca_college)
pca_college$rotation[, 1:3]  # Loadings za prvih 3 PC

# Scree plot
var_c <- pca_college$sdev^2
pve_c <- var_c / sum(var_c)
plot(pve_c, type = "b", pch = 16,
     xlab = "PC", ylab = "PVE",
     main = "Scree plot – College")

# Vizualizacija prvih dviju komponenti
scores <- as.data.frame(pca_college$x)
plot(scores$PC1, scores$PC2,
     col = ifelse(College$Private == "Yes", "blue3", "red3"),
     pch = 16, cex = 0.7,
     xlab = "PC1", ylab = "PC2",
     main = "PCA – College (plava = Private, crvena = Public)")
legend("topright", legend = c("Privatni", "Javni"),
       col = c("blue3", "red3"), pch = 16, bty = "n")
```

### Interpretacija loadinga – kako prepoznati što PC mjeri

```r
# Gledamo loadinge prvih PC
pca_us$rotation[, 1:2]

# PC1 za USArrests:
# Murder:   -0.536
# Assault:  -0.583
# UrbanPop: -0.278
# Rape:     -0.543
# → PC1 je "opći kriminalitet" – visoke pozitivne vrijednosti = manje kriminala

# PC2:
# Murder:   -0.418
# Assault:  -0.188
# UrbanPop:  0.873
# Rape:      0.167
# → PC2 dominira urbanizacija – odvojena od kriminala
```

## Interpretacija rezultata

**PVE:**
> "Prva glavna komponenta objašnjava 62.0% ukupne varijabilnosti podataka. Prve dvije komponente zajedno objašnjavaju 87.0% ukupne varijabilnosti, što upućuje da se višedimenzionalni skup podataka može dobro aproksimirati u dvodimenzionalnom prostoru."

**Loadings:**
> "PC1 ima visoka negativna opterećenja za varijable `Murder` (-0.536), `Assault` (-0.583) i `Rape` (-0.543), dok je opterećenje za `UrbanPop` manje (-0.278). PC1 možemo interpretirati kao opći indeks kriminala – savezne države s visokim vrijednostima PC1 (u negativnom smjeru) imaju višu stopu kriminaliteta."

**Biplot:**
> "U biplotu varijable `Murder`, `Assault` i `Rape` imaju strelice usmjerene u sličnom smjeru, što indicira pozitivnu korelaciju. Varijabla `UrbanPop` je usmjerena u drugom smjeru, što sugerira da urbanizacija nije snažno vezana uz nasilje u ovom skupu podataka."

**Scree plot:**
> "Scree plot pokazuje 'lakat' (elbow) nakon druge komponente – PVE pada naglo do PC2, a zatim se stabilizira. To sugerira da su prve dvije komponente dovoljne za reprezentaciju podataka."

## Najčešće greške

- **Zaboraviš `scale = TRUE`** – ako varijable imaju različite mjerne jedinice, PCA bez standardizacije daje rezultate koji su dominantni varijablama s većim vrijednostima
- **Ignoriraš predznak loadinga** – sam predznak nema apsolutno značenje, ali relativni predznaci (+ vs –) su informativni
- **Tražiš "pravi" broj komponenti** – nema automatskog pravila; scree plot i kumulativni PVE su vodiči (tipično 70–80% varijance)
- **Zabrkaš loadings i scores** – loadings su koeficijenti (rotacija), scores su koordinate opažanja u novom prostoru

## Mini podsjetnik – PCA

| Što trebam | R kod |
|---|---|
| Provesti PCA | `prcomp(data, scale = TRUE)` |
| Pregled varijance | `summary(pca_obj)` |
| Loadings | `pca_obj$rotation` |
| Scores | `pca_obj$x` |
| Scree plot | `plot(pca_obj$sdev^2 / sum(pca_obj$sdev^2), type="b")` |
| Biplot | `biplot(pca_obj)` |

---

# 16. Završni globalni sažetak

## Tablica: Kada koristiti koju metodu?

| Odzivna varijabla | Problem | Preporučena metoda |
|---|---|---|
| Kontinuirana (numerička) | Regresija | Linearna regresija, Random Forest (regresija) |
| Binarna (0/1, Da/Ne) | Klasifikacija | Logistička regresija, Stabla, RF, SVM |
| Više klasa | Klasifikacija | Multinomna logistička regresija, RF, SVM |
| Brojevna (0,1,2,...) | Regresija broja događaja | Poissonova regresija, quasipoisson |
| — | Redukcija dimenzionalnosti | PCA |
| — | Procjena greške modela | Cross-validacija |
| — | Procjena SE procjenitelja | Bootstrap |

## Ključne R funkcije po temama

```r
# LINEARNA REGRESIJA
lm(Y ~ X, data = df)                     # model
summary(model)                            # sažetak
predict(model, newdata, interval = "...")  # predikcija
AIC(m1, m2); BIC(m1, m2)                 # usporedba
anova(m1, m2)                             # F-test

# DIJAGNOSTIKA
par(mfrow = c(2,2)); plot(model)          # 4 dijagnostička grafa
shapiro.test(residuals(model))            # normalnost
bptest(model)                             # homoskedastičnost
vif(model)                                # kolinearnost

# LOGISTIČKA REGRESIJA
glm(Y ~ X, family = binomial, data = df)  # model
exp(coef(model))                          # odds ratios
predict(model, type = "response")         # vjerojatnosti
table(pred, stvarno)                      # matrica zabune

# POISSONOVA REGRESIJA
glm(Y ~ X, family = poisson, data = df)  # model
glm(Y ~ X, family = quasipoisson)        # za prekomjernu disperziju
deviance(m) / df.residual(m)             # provjera disperzije
exp(coef(model))                          # multiplikativni efekti

# CROSS-VALIDACIJA
library(boot)
cv.glm(data, glm_model)$delta[1]         # LOOCV
cv.glm(data, glm_model, K=10)$delta[1]  # k-fold CV

# BOOTSTRAP
boot(data, statistic_fun, R = 1000)      # bootstrap
boot.ci(boot_obj, type = "perc")         # IP

# STABLA
library(tree)
tree(Y ~ ., data = train)                # stablo
cv.tree(stablo, FUN = prune.misclass)    # CV za pruning
prune.misclass(stablo, best = k)         # rezanje

# RANDOM FOREST
library(randomForest)
randomForest(Y ~ ., data, mtry = m, ntree = 500, importance = TRUE)
importance(rf_model)                      # važnost varijabli
varImpPlot(rf_model)                      # grafički

# SVM
library(e1071)
svm(Y ~ ., data, kernel = "linear", cost = 1)
svm(Y ~ ., data, kernel = "radial", cost = 1, gamma = 0.1)
tune(svm, Y ~ ., data, ranges = list(cost = ..., gamma = ...))

# PCA
prcomp(data, scale = TRUE)               # PCA
summary(pca_obj)                          # PVE
pca_obj$rotation                         # loadings
pca_obj$x                                # scores
biplot(pca_obj)                           # biplot
```

## Ključne interpretacijske rečenice

### Linearna regresija
> "Koeficijent uz `X` iznosi β̂ i statistički je značajan (p < 0.05). Povećanje `X` za 1 jedinicu, uz sve ostale prediktore nepromijenjene, mijenja Y za β̂ jedinica."

> "Prilagođeni R² = 0.XX znači da model objašnjava XX% varijabilnosti odzivne varijable, uz korekciju za broj prediktora."

### Logistička regresija
> "Odds ratio za prediktor `X` iznosi e^β̂. Povećanje `X` za 1 množi izglede pozitivnog ishoda s e^β̂."

> "Model postiže točnost XX%, osjetljivost XX% i specifičnost XX% na testnom skupu."

### Poissonova regresija
> "Koeficijent uz `X` iznosi β̂. Povećanje `X` za 1 jedinicu množi očekivani broj događaja s e^β̂ = X.XX, uz ostale prediktore fiksirane."

> "Omjer devijance i stupnjeva slobode = X.X >> 1, što upućuje na prekomjernu disperziju. Quasipoisson model daje konzervativnije procjene SE."

### PCA
> "Prve dvije glavne komponente zajedno objašnjavaju XX% ukupne varijabilnosti. PC1 možemo interpretirati kao [opis na temelju loadinga]."

> "Na biplotu varijable A i B imaju strelice u sličnom smjeru, što upućuje na pozitivnu korelaciju, dok varijabla C pokazuje suprotan smjer."

### CV i Bootstrap
> "10-fold cross-validacija pokazuje da model M2 (CV MSE = X.XX) ima manju procijenjenu testnu grešku od modela M1 (CV MSE = Y.YY). Odabiremo M2 kao prikladniji model."

> "Bootstrap SE koeficijenta iznosi X.XX, što je konzistentno s teorijskom SE = X.XX iz summary() i potvrđuje stabilnost procjene."

---

## Checklist za rješavanje zadataka na ispitu

### Prije analize
- [ ] Učitao sam podatke i provjerio strukturu (`str()`, `summary()`, `head()`)
- [ ] Provjera tipa odzivne varijable → biramo prikladnu metodu
- [ ] Vizualizacija distribucija i veza (`pairs()`, `boxplot()`, `plot()`)
- [ ] Provjera nedostajućih vrijednosti (`sum(is.na(data))`)

### Linearna regresija
- [ ] Izgradio model (`lm()`)
- [ ] Interpretirao `summary()` – koeficijenti, p-vrijednosti, R²
- [ ] Napravio dijagnostiku (`plot(model)`, `bptest()`, `shapiro.test()`)
- [ ] Provjerio VIF ako ima više prediktora (`vif()`)
- [ ] Usporedio modele (AIC ili `anova()`)

### Logistička regresija
- [ ] `glm(..., family = binomial)`
- [ ] Interpretirao koeficijente i odds ratios (`exp(coef())`)
- [ ] Predikcija: `predict(..., type = "response")` – OBAVEZNO!
- [ ] Klasificirao s odabranim pragom (`ifelse(vjer > 0.5, ...)`)
- [ ] Izračunao matricu zabune i mjere (točnost, osjetljivost, specifičnost)

### Poissonova regresija
- [ ] `glm(..., family = poisson)`
- [ ] Provjerio prekomjernu disperziju (`deviance/df.residual`)
- [ ] Ako je disperzija > 2: quasipoisson
- [ ] Interpretirao koeficijente multiplikativno (`exp(coef())`)

### Klasifikacija (stablo, RF, SVM)
- [ ] Podijelio na trening/testni skup (`set.seed()` i `sample()`)
- [ ] Izgradio model na trening skupu
- [ ] Evaluirao na testnom skupu (matrica zabune + mjere)
- [ ] Usporedio više metoda

### PCA
- [ ] Standardizirao podatke (`scale = TRUE`)
- [ ] Provjerio PVE i kumulativni PVE (`summary()`)
- [ ] Interpretirao loadinge prvih PC
- [ ] Nacrtao scree plot i biplot
- [ ] Zaključio koliko PC je dovoljno (scree + 70-80% PVE)

### CV i Bootstrap
- [ ] Za odabir modela: CV (`cv.glm()`)
- [ ] Za SE procjenitelja: bootstrap (`boot()`)
- [ ] Koristio `set.seed()` za reproducibilnost

---

## Tipične pogreške koje trebam izbjegavati

| Greška | Ispravan postupak |
|---|---|
| `predict()` bez `type = "response"` za glm | Uvijek navesti `type = "response"` za vjerojatnosti |
| Standardizirati test skup posebno (ne s trening parametrima) | Uvijek standardizirati test s `center` i `scale` iz trening skupa |
| Uspoređivati R² direktno za odabir modela | Koristiti prilagođeni R², AIC ili CV |
| Visoka točnost = dobar model kod rijetkih klasa | Pratiti osjetljivost i specifičnost odvojeno |
| `family = binomial` za Poissonov model | `family = poisson` za brojevne podatke |
| `scale = FALSE` u PCA kad su varijable na raznim skalama | Uvijek `scale = TRUE` osim ako su varijable standardizirane |
| Zamijeniti loadings i scores | Loadings = `$rotation` (varijable), Scores = `$x` (opažanja) |
| Prenaučenost – ne evaluirati na testnom skupu | Uvijek procijeniti testnu grešku; nikad samo trening |

---
