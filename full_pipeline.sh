

# 0 - Initial setup
mkdir -p corpora
mkdir -p data/working
mkdir -p data/external
mkdir -p results
mkdir -p logs
chmod -R +x src
# sudo apt-get update
sudo apt-get install pandoc # to execute RMarkdown

# 1 - Download wiki dump
WIKI_URL=https://archive.org/download/enwiki-20210401
WIKI_FILE=enwiki-20210401-pages-articles.xml.bz2
wget -c -b -P corpora/ $WIKI_URL/$WIKI_FILE
    # "-c": continue getting a partially-downloaded file
    # "-b": go to background after startup. Output is redirected to wget-log.

# 2 - Extract corpus into a raw .txt file
src/data/extract_wiki_dump.sh corpora/enwiki-20210401-pages-articles.xml.bz2

# 3 - Create text file with one line per sentence and removing paragraphs of less than 50 words
python3 -u src/data/tokenize_and_reduce_corpus.py corpora/enwiki-20210401-pages-articles.txt

# 4 - Clean corpus
CORPUS_IN=corpora/enwiki-20210401-pages-articles_sentences.txt &&
CORPUS_OUT=corpora/wiki2021.txt &&
src/data/clean_corpus.sh $CORPUS_IN > $CORPUS_OUT
# check number of lines,words,characters with:
wc corpora/wiki2021.txt
# 78051838  1749313740 10453079770 corpora/wiki2021.txt

# 6 - Shuffle corpus. Set seed in `src/data/shuffle_corpus_multiple.sh`.
    # The new corpus is named as `corpora/{CORPUS_ID}s<seed>.txt`. 
CORPUS_ID=wiki2021 &&
src/data/shuffle_corpus_multiple.sh $CORPUS_ID

# 7 - Get vocabulary using GloVe module
OUT_DIR=data/working &&
VOCAB_MINCOUNT=100 &&
IDS=(wiki2021) &&
for id in ${IDS[@]}; do
    corpus=corpora/$id.txt
    src/corpus2vocab.sh $corpus $OUT_DIR $VOCAB_MINCOUNT
done

# 8 - Resample corpus to achieve target frequencies in word B and create vocab. of new corpora
resample_corpus.sh "she" "he" &&
resample_corpus.sh "he" "she" &&
resample_corpus.sh "african" "european" &&
resample_corpus.sh "rich" "poor"

# 9 - Train word embeddings in all corpora
nohup src/corpus2sgns_multiple.sh &> logs/nohup_sgns.out &
nohup src/corpus2fasttext_multiple.sh &> logs/nohup_fasttext.out &
nohup src/corpus2glove_multiple.sh &> logs/nohup_glove.out &

# 10 - Run PCA on normalized vectors of unshuffled and shuffled corpus
python src/matrices2pca.py --normalize

# 11 - Run hyperparameter trials
nohup src/embeddings_hyperparams.sh &> logs/nohup_hyperparams.out &

# 12 - Compute bias wrt context words (definir files en el sh)
A="SHE" && B="HE" && src/biasdf_multiple.sh $A $B
A="HE" && B="SHE" && src/biasdf_multiple.sh $A $B
A="AFRICAN" && B="EUROPEAN" && src/biasdf_multiple.sh $A $B
A="RICH" && B="POOR" && src/biasdf_multiple.sh $A $B

# 13 - Figures of cosine similarities heatmaps
jupyter nbconvert --to html --execute notebooks/similarities_sgns.ipynb
jupyter nbconvert --to html --execute notebooks/similarities_fasttext.ipynb
jupyter nbconvert --to html --execute notebooks/similarities_glove.ipynb

# 14 - Figures of bias (resampling experiment), hyperparameters and PCA
    # Download glasgow norms:
wget -O data/external/GlasgowNorms.csv https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-018-1099-3/MediaObjects/13428_2018_1099_MOESM2_ESM.csv 
R -e 'rmarkdown::render("notebooks/bias_resampling_gender.Rmd", "html_document")' &&
R -e 'rmarkdown::render("notebooks/bias_resampling_gender_she.Rmd", "html_document")' &&
R -e 'rmarkdown::render("notebooks/bias_resampling_ethnicity.Rmd", "html_document")' && 
R -e 'rmarkdown::render("notebooks/bias_resampling_affluence.Rmd", "html_document")' && 
R -e 'rmarkdown::render("notebooks/metrics_hyperparams.Rmd", "html_document")' && 
R -e 'rmarkdown::render("notebooks/pca_normalized.Rmd", "html_document")'

# 15 - Make grids of figures for paper
python src/make_plots_grids.py &&
Rscript src/bias_resampling_grid.R
