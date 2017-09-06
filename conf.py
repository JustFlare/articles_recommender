mode = 'fit'  # обучение и ранжирование для всех данных, update - дообучение уже сохраненный модели и rank - ранжировать с помощью уже сохраненной модели
saved_model = ""  # путь к файлу сохраненной модели для режимов update или rank
algorithm = "doc2vec"  # может быть lda или doc2vec
do_lemmatize = False  # делать ли вообще лемматизацию. может быть True или False
lemmatized_data = ""  # путь к лемматизированной выборке. если нужно лемматизировать выборку заново - оставьте этот параметр пустым
data_dir = 'chelb'  # путь к папке с данными (может быть любое количество подпапок). ВАЖНО: файлы должны быть пронумерованы уникальными индексами
data_encoding = "utf-8"  # кодировка файлов в папке, например utf-8, cp1251
num_best = 5  # количество наиболее похожих документов в выдаче
only_title = True # если стоит в True - то из текстов в алгоритм попадают только заголовки

# параметры алгоритмов. я выделил наиболее важные на свой взгляд, полные списки и доки:
# doc2vc: https://radimrehurek.com/gensim/models/doc2vec.html
# lda: https://radimrehurek.com/gensim/models/ldamodel.html

# doc2vec:
n_epochs = 50  # количество итераций при обучении модели
vector_dim = 100  # размерность вектора для каждого слова
window = 5  # сколько соседних слов учитывать (в обе стороны)
alpha = 0.025  # скорость обучения при градиентном спуске, будет снижаться до min_alpha
min_count = 0  # игнорировать все слова с частотой ниже заданной

# lda:
topics = 15   # количество топиков (тем)
passes = 1  # количество полных прогонов обучения (подробней тут: https://groups.google.com/forum/#!topic/gensim/ojySenxQHi4)
iterations = 50  # ограничение на количество раз, которое алгоритм будет вызывать Expectation step в EM алгоритме
eval_every = 10  # каждую eval_every условную итерацию оценивать модель через расчет перплексии (подробней см. здесь: http://www.machinelearning.ru/wiki/images/0/0c/NizhibitskyLomonosovSlides14.pdf)
min_prob = 0.01  # вероятностный фильтер топиков (топики с меньшей вероятностью отсекаются)
