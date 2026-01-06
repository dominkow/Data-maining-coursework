#__________________________________________________________________________________________
#__________________________________________________________________________________________
#---------- KLASYFIKACJA  na przykładzie metody k-NN (k-Nearest Neighbors)  ---------------
#
# Kurs: Eksploracja danych
# Materiały pomocnicze na laboratorium
# (C) A.Zagdański
#__________________________________________________________________________________________
#__________________________________________________________________________________________


#__________________________________________________________________________________________
#--------------------- 1 Dane wykorzystane w przykładzie ----------------------------------
#__________________________________________________________________________________________

# zilustrujemy zastosowanie metody k-NN na przykładzie danych dotyczących irysów
library(MASS)
data(iris)

# Przed budową modeli klasyfikacyjnych powinniśmy przyjrzeć się danym (analiza opisowa),
# zwracając uwagę m.in. na ich charakterystyczne własności oraz próbując (wstępnie) ocenić
# zdolności dyskryminacyjne (predykcyjne) poszczególnych zmiennych/cech.
#
# W szczególności możemy wykorzystać w tym celu (zwykłe i skategoryzowane):
#  -- wykresy pudełkowe,
#  -- histogramy,
#  -- wykresy rozrzutu,
#  -- ...


#__________________________________________________________________________________________
#---------------------- 2 Zastosowanie funkcji knn() z pakietu class ----------------------
#__________________________________________________________________________________________

library(class)

# losowanie podzbiorów
n <- dim(iris)[1]

# losujemy obiekty do zbioru uczącego i testowego
learning.set.index <- sample(1:n,2/3*n)

# tworzymy zbiór uczący i testowy
learning.set <- iris[learning.set.index,]
test.set     <- iris[-learning.set.index,]

# rzeczywiste gatunki irysów
etykietki.rzecz <- test.set$Species

# teraz robimy prognozę
etykietki.prog <- knn(learning.set[,-5], test.set[,-5], learning.set$Species, k=5)

# macierz pomyłek (ang. confusion matrix)
(wynik.tablica <- table(etykietki.prog,etykietki.rzecz))

# błąd klasyfikacji
n.test <- dim(test.set)[1]
(n.test - sum(diag(wynik.tablica))) / n.test

# Wady funkcji knn(){class}:
#  -- trudno jest wybierać cechy,
#  -- w R zazwyczaj modele klasyfikacyjne buduje się inaczej, wykorzystując tzw. język formuł.
#	Łatwiej wtedy widać, na czym polega budowa i ocena jakości modeli klasyfikacyjnych.


#__________________________________________________________________________________________
#------------------- 3 Zastosowanie funkcji ipredknn() z pakietu ipred --------------------
#__________________________________________________________________________________________


# Uwaga:
# Implementacja w pakiecie "ipred" wykorzystuje tzw. język formuł
# i jest ona znacznie bardziej zbliżona jest do standardowych implementacji
# modeli klasyfikacyjnych (modeli predykcyjnych) w systemie R.

library(ipred)

# budujemy model
model.knn.1 <- ipredknn(Species ~ ., data=learning.set, k=5)

# zobaczmy, co jest w środku
model.knn.1
summary(model.knn.1)
attributes(model.knn.1)

# sprawdźmy jakość modelu
# uwaga: czasami funkcje "predict" działają niestandardowo
etykietki.prog <- predict(model.knn.1, test.set, type="class")
etykietki.prog

# macierz pomyłek (ang. confusion matrix)
(wynik.tablica <- table(etykietki.prog,etykietki.rzecz))

# błąd klasyfikacji
(n.test - sum(diag(wynik.tablica))) / n.test

# Można również skonstruować modele oparte na innych zmiennych oraz innej liczbie sąsiadów
# i sprawdzić/porównać ich skuteczność.

# Przykłady innych modeli:
model.knn.1a <- ipredknn(Species ~ ., data=learning.set, k=1)
model.knn.1b <- ipredknn(Species ~ ., data=learning.set, k=15)
# Petal.Length + Petal.Width (cechy o "lepszych" zdolnościach dyskryminacyjnych)
model.knn.2  <- ipredknn(Species ~ Petal.Length + Petal.Width, data=learning.set, k=5)
model.knn.2a <- ipredknn(Species ~ Petal.Length + Petal.Width, data=learning.set, k=1)
model.knn.2b <- ipredknn(Species ~ Petal.Length + Petal.Width, data=learning.set, k=15)
# Sepal.Length + Sepal.Width (cechy o "gorszych" zdolnościach dyskryminacyjnych)
model.knn.3  <- ipredknn(Species ~ Sepal.Length + Sepal.Width, data=learning.set, k=5)
model.knn.3a <- ipredknn(Species ~ Sepal.Length + Sepal.Width, data=learning.set, k=1)
model.knn.3b <- ipredknn(Species ~ Sepal.Length + Sepal.Width, data=learning.set, k=15)


#__________________________________________________________________________________________
#------------ 4 Inne (zaawansowane) sposoby oceny dokładności klasyfikacji ----------------
#__________________________________________________________________________________________

# Często zamiast jednokrotnego podziału na zbiór uczący stosuje się metodę cross-validation 
# polegającą na wielokrotnym losowaniu zbioru uczącego i testowego, 
# budowie klasyfikatora na zbiorze uczącym, sprawdzenia go na testowym oraz uśrednieniu wyników.
# Oczywiście, da się zrobić wszystko "na piechotę" w pętli i uśrednić, ale można wykorzystać gotowe implementacje.

library(ipred)

# żeby skorzystać z gotowych funkcji należy przygotować sobie "wrapper" dostosowujący
# funkcję "predict" dla naszego modelu do standardu wymaganego przez "errorest"

my.predict  <- function(model, newdata) predict(model, newdata=newdata, type="class")
my.ipredknn <- function(formula1, data1, ile.sasiadow) ipredknn(formula=formula1,data=data1,k=ile.sasiadow)

# porównanie błędów klasyfikacji: cv, boot, .632plus
errorest(Species ~., iris, model=my.ipredknn, predict=my.predict, estimator="cv",     est.para=control.errorest(k = 10), ile.sasiadow=5)
errorest(Species ~., iris, model=my.ipredknn, predict=my.predict, estimator="boot",   est.para=control.errorest(nboot = 50), ile.sasiadow=5)
errorest(Species ~., iris, model=my.ipredknn, predict=my.predict, estimator="632plus",est.para=control.errorest(nboot = 50), ile.sasiadow=5)

# Eksperyment: badamy wpływ liczby sąsiadów na skuteczność modelu
liczba.sasiadow.zakres <- 1:15
wyniki1 <-  sapply(liczba.sasiadow.zakres, function(k)
errorest(Species ~., iris, model=my.ipredknn, predict=my.predict, estimator="cv", est.para=control.errorest(k=10), ile.sasiadow=k)$error)
plot(liczba.sasiadow.zakres, type="b", wyniki1, lwd=2, main="wpływ liczby sąsiadów na błąd klasyfikacji", xlab="k (liczba sąsiadów)", ylab="błąd klasyfikacji")

#__________________________________________________________________________________________
#----------------- 5 Obszary decyzyjne (dla wybranych par zmiennych) ----------------------
#__________________________________________________________________________________________

library(klaR)

# dla ułatwienia zmieniamy nazwy klas na "A", "B" i "C"
levels(iris$Species) = c("A","B","C") # A-"setosa"  B-"versicolor" C-"virginica"

drawparti( iris$Species, x=iris$Petal.Length, y=iris$Petal.Width, method = "sknn", k=1,  xlab="PL",ylab="PW")
drawparti( iris$Species, x=iris$Petal.Length, y=iris$Petal.Width, method = "sknn", k=15,  xlab="PL",ylab="PW")

partimat(Species ~ ., data = iris, method = "sknn",  plot.matrix = FALSE, imageplot = TRUE)
