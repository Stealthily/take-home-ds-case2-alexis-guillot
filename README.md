# Explanation of deliverable and video call

Welcome to our take home data science case! We're very happy that you applied for our new role.
Today you will be given a challenge for which we want you to find a solution.
You're free to be creative and use any package or method.

After 24 hours you have to make a pull request, containing at least the following:

1. A working & understandable code solving the case
2. A requirements.txt file with packages used
3. A short presentation explaining how and why you've chosen this method / solution,
   how your solution performs and which recommendations you have for further improvements if more time would be available.

After 24 hours we will have a video call to walk through your solution, chosen methods, performance and recommendations. You will use the presentation as means to communicate clearly, this is something we often do in our company to keep everything understandable and explainable.

# Background information
KeyWI is an ai-driven solution for online marketeers. We provide valuable data insights for keywords and help SEO specialists to make the best choices. Therefor, we cluster keywords, predict search intention, show current rank, and much more.

We are always interested in what does work and what does not work. Let me explain, some websites do rank, while others don't.. how come? To be able to aid in the creation of new content, we want to provide insights in currently ranking content. That's why we want to show which type of page rank, what kind of content did people create?

# How to get started?

To start the case clone this repo to your machine and switch to a new branch.
- [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the repo to your own github account
- `git clone <reponame>`

After cloning, you should download a file called `data.zip`, which contains a `pages` folder with png and html files and a `labels.csv`.
which is publicly available on `https://keywi-public.s3.eu-west-1.amazonaws.com/data.zip`.

`labels.csv` is a csv file containing two columns, `hash` and `label`. For almost
each hash (some might be missing) there is a scraped html page and a screenshot in `pages` named `{HASH}.html` and `{HASH}.png` respectively.

# Label explanation

There are different labels
- landing: A landing or home page of a website. e.g. `https://www.bulletcoffeeco.com/`
- social: A post or page on a social media platform. e.g. `https://www.pinterest.com/pin/468726273693073815/` or `https://www.facebook.com/digitalestad/`
- article: An article or blog post, e.g. `https://nl.wikipedia.org/wiki/Martin_Schulz` or `https://www.sumologic.com/blog/data-tagging-classification-enrichment/`
- commercial: A overview or product page for an online webshop, e.g. `https://www.amazon.nl/IKEA-Sortera-Afvalemmer-deksel-plastic/dp/B00GMMBZAO` or `https://www.bol.com/nl/nl/l/tv-meubels/14088/`

# Data caveats

The labels are given based on a set of rules applied to the url of the webpage, so there may be mistakes in the labels of the data. However, almost all the labels will be correct.

During data collection a couple of problems may have occurred, you will see this back in the data.
Examples of this are a cookie modal, redirect to a login page, or a 404 error.

# Task

**It is your task to apply machine learning techniques to create a model which can predict unseen scraped web pages.**

- Your model should be able to predict the page type of an unseen html - png combination.
- You are free to use any additional publicly and legally available data sources to perform this task.
- The model is not expected to be perfect, we understand 24 hours is a short time frame. 

If you have a question or something else is unclear, please let us know via e-mail. We'll try to respond as fast as possible.
