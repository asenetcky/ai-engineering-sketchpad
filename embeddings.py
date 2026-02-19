# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "openai==2.21.0",
#     "polars==1.38.1",
#     "python-dotenv==1.2.1",
#     "scikit-learn==1.8.0",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from openai import OpenAI
    import json
    from dotenv import load_dotenv
    import os

    return OpenAI, load_dotenv, os


@app.cell
def _(OpenAI, load_dotenv, os):
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_TOKEN")

    client = OpenAI(api_key=OPENAI_API_KEY)
    return (client,)


@app.cell
def _(client):
    _response = client.embeddings.create(
        model="text-embedding-3-small",
        input="Embeddings are a numerical representation of text used to measure the relatedness between pieces of text."
    )
    response_dict=_response.model_dump()
    return (response_dict,)


@app.cell
def _(response_dict):
    response_dict
    return


@app.cell
def _():
    products=[{'title': 'Smartphone X1',
      'short_description': 'The latest flagship smartphone with AI-powered features and 5G connectivity.',
      'price': 799.99,
      'category': 'Electronics',
      'features': ['6.5-inch AMOLED display',
       'Quad-camera system with 48MP main sensor',
       'Face recognition and fingerprint sensor',
       'Fast wireless charging']},
     {'title': 'Luxury Diamond Necklace',
      'short_description': 'Elegant necklace featuring genuine diamonds, perfect for special occasions.',
      'price': 1499.99,
      'category': 'Beauty',
      'features': ['18k white gold chain',
       '0.5 carat diamond pendant',
       'Adjustable chain length',
       'Gift box included']},
     {'title': 'RC Racing Car',
      'short_description': 'High-speed remote-controlled racing car for adrenaline-packed fun.',
      'price': 89.99,
      'category': 'Toys',
      'features': ['Top speed of 30 mph',
       'Responsive remote control',
       'Rechargeable battery',
       'Durable construction']},
     {'title': 'Ultra HD 4K TV',
      'short_description': 'Immerse yourself in stunning visuals with this 65-inch 4K TV.',
      'price': 1299.99,
      'category': 'Electronics',
      'features': ['65-inch 4K UHD display',
       'Dolby Vision and HDR10+ support',
       'Smart TV with streaming apps',
       'Voice remote included']},
     {'title': 'Glowing Skin Serum',
      'short_description': 'Revitalize your skin with this nourishing serum for a radiant glow.',
      'price': 39.99,
      'category': 'Beauty',
      'features': ['Hyaluronic acid and vitamin C',
       'Hydrates and reduces fine lines',
       'Suitable for all skin types',
       'Cruelty-free']},

     {'title': 'LEGO Space Shuttle',
      'short_description': 'Build your own space adventure with this LEGO space shuttle set.',
      'price': 49.99,
      'category': 'Toys',
      'features': ['359 pieces for creative building',
       'Astronaut minifigure included',
       'Compatible with other LEGO sets',
       'For ages 7+']},
     {'title': 'Wireless Noise-Canceling Headphones',
      'short_description': 'Enjoy immersive audio and block out distractions with these headphones.',
      'price': 199.99,
      'category': 'Electronics',
      'features': ['Active noise cancellation',
       'Bluetooth 5.0 connectivity',
       'Long-lasting battery life',
       'Foldable design for portability']},
     {'title': 'Luxury Perfume Gift Set',
      'short_description': 'Indulge in a collection of premium fragrances with this gift set.',
      'price': 129.99,
      'category': 'Beauty',
      'features': ['Five unique scents',
       'Elegant packaging',
       'Perfect gift for fragrance enthusiasts',
       'Variety of fragrance notes']},
     {'title': 'Remote-Controlled Drone',
      'short_description': 'Take to the skies and capture stunning aerial footage with this drone.',
      'price': 299.99,
      'category': 'Electronics',
      'features': ['4K camera with gimbal stabilization',
       'GPS-assisted flight',
       'Remote control with smartphone app',
       'Return-to-home function']},
     {'title': 'Luxurious Spa Gift Basket',
      'short_description': 'Pamper yourself or a loved one with this spa gift basket full of relaxation goodies.',
      'price': 79.99,
      'category': 'Beauty',
      'features': ['Bath bombs, body lotion, and more',
       'Aromatherapy candles',
       'Reusable wicker basket',
       'Great for self-care']},
     {'title': 'Robot Building Kit',
      'short_description': 'Learn robotics and coding with this educational robot building kit.',
      'price': 59.99,
      'category': 'Toys',
      'features': ['Build and program your own robot',
       'STEM learning tool',
       'Compatible with Scratch and Python',
       'Ideal for young inventors']},
     {'title': 'High-Performance Gaming Laptop',
      'short_description': 'Dominate the gaming world with this powerful gaming laptop.',
      'price': 1499.99,
      'category': 'Electronics',
      'features': ['Intel Core i7 processor',
       'NVIDIA RTX graphics',
       '144Hz refresh rate display',
       'RGB backlit keyboard']},
     {'title': 'Natural Mineral Makeup Set',
      'short_description': 'Enhance your beauty with this mineral makeup set for a flawless look.',
      'price': 34.99,
      'category': 'Beauty',
      'features': ['Mineral foundation and eyeshadows',
       'Non-comedogenic and paraben-free',
       'Cruelty-free and vegan',
       'Includes makeup brushes']},
     {'title': 'Interactive Robot Pet',
      'short_description': 'Adopt your own robot pet that responds to your voice and touch.',
      'price': 79.99,
      'category': 'Toys',
      'features': ['Realistic pet behaviors',
       'Voice recognition and touch sensors',
       'Teaches responsibility and empathy',
       'Rechargeable battery']},
     {'title': 'Smart Thermostat',
      'short_description': "Control your home's temperature and save energy with this smart thermostat.",
      'price': 129.99,
      'category': 'Electronics',
      'features': ['Wi-Fi connectivity',
       'Energy-saving features',
       'Compatible with voice assistants',
       'Easy installation']},
     {'title': 'Designer Makeup Brush Set',
      'short_description': 'Upgrade your makeup routine with this premium designer brush set.',
      'price': 59.99,
      'category': 'Beauty',
      'features': ['High-quality synthetic bristles',
       'Chic designer brush handles',
       'Complete set for all makeup needs',
       'Includes stylish carrying case']},
     {'title': 'Remote-Controlled Dinosaur Toy',
      'short_description': 'Roar into action with this remote-controlled dinosaur toy with lifelike movements.',
      'price': 49.99,
      'category': 'Toys',
      'features': ['Realistic dinosaur sound effects',
       'Walks and roars like a real dinosaur',
       'Remote control included',
       'Educational and entertaining']},
     {'title': 'Wireless Charging Dock',
      'short_description': 'Charge your devices conveniently with this sleek wireless charging dock.',
      'price': 39.99,
      'category': 'Electronics',
      'features': ['Qi wireless charging technology',
       'Supports multiple devices',
       'LED charging indicators',
       'Compact and stylish design']},
     {'title': 'Luxury Skincare Set',
      'short_description': 'Elevate your skincare routine with this luxurious skincare set.',
      'price': 179.99,
      'category': 'Beauty',
      'features': ['Premium anti-aging ingredients',
       'Hydrating and rejuvenating formulas',
       'Complete skincare regimen',
       'Elegant packaging']}]
    return (products,)


@app.cell
def _(client, products):
    product_text = [product["short_description"] for product in products]
    # print(product_text)

    _response = client.embeddings.create(
        model="text-embedding-3-small",
        input=product_text
    )

    _response_dict= _response.model_dump()
    print(_response_dict)

    for i, product in enumerate(products):
        product["embedding"] = _response_dict["data"][i]["embedding"]

    print(products[0].items())
    return


@app.cell
def _(products):
    products[0]
    return


@app.cell
def _(products):
    # t sne
    from sklearn.manifold import TSNE
    import numpy as np

    categories = [product["category"] for product in products]
    embeddings = [product["embedding"] for product in products]

    # reduce the number of embedding dimensions to two

    tsne = TSNE(n_components=2, perplexity=5)
    embeddings_2d = tsne.fit_transform(np.array(embeddings))
    return embeddings_2d, np


@app.cell
def _(embeddings_2d):
    embeddings_2d
    return


@app.cell
def _(embeddings_2d, products):
    # visualizing embedded descriptions
    import matplotlib.pyplot as plt

    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    topics = [product["category"] for product in products]
    for _i, topic in enumerate(topics):
        plt.annotate(topic, (embeddings_2d[_i, 0], embeddings_2d[_i, 1]))

    plt.show()
    return


@app.cell
def _(client):
    def create_embeddings(texts):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )

        response_dict = response.model_dump()

        return [data["embedding"] for data in response_dict["data"]]

    return (create_embeddings,)


@app.cell
def _(create_embeddings):
    create_embeddings("hola! I'm a caterpiller!")
    return


@app.cell
def _(create_embeddings):
    create_embeddings(["I like polars!", "I like duckplyr!", "Why can't we all just get along?"])
    return


@app.cell
def _(create_embeddings, np, products):
    from scipy.spatial import distance

    search_text = "music"
    search_embedding = create_embeddings(search_text)[0]

    # cosine distance is a popular choice as a distance metric for identifying semantically similar texts.

    distances = []
    for _product in products:
        dist = distance.cosine(search_embedding, _product["embedding"])
        distances.append(dist)

    min_dist_ind = np.argmin(distances)
    return distance, min_dist_ind


@app.cell
def _(min_dist_ind, products):
    print(products[min_dist_ind]['title'])
    print(products[min_dist_ind]['short_description'])
    return


@app.cell
def _(products):
    products[0]
    return


@app.cell
def _(products):
    # Define a function to combine the relevant features into a single string
    def create_product_text(product):
      return f"""Title: {product["title"]}
        Description: {product["short_description"]}
        Category: {product["category"]}
        Features: {', '.join(product["features"])}"""

    product_texts = [create_product_text(product) for product in products]
    product_texts

    return create_product_text, product_texts


@app.cell
def _(create_embeddings, product_texts):
    product_embeddings = create_embeddings(product_texts)

    return (product_embeddings,)


@app.cell
def _(distance):
    def find_n_closest(query_vector, embeddings, n=3):
        distances = []

        for index, embedding in enumerate(embeddings):
            dist = distance.cosine(query_vector, embedding)
            distances.append({"distance": dist, "index":index})
        
        distances_sorted = sorted(distances, key=lambda x: x["distance"])
        return distances_sorted[0:n]

    return (find_n_closest,)


@app.cell
def _(create_embeddings, find_n_closest, product_embeddings, products):

    _query_text = "charger"
    _query_vector = create_embeddings(_query_text)[0]

    _hits=find_n_closest(_query_vector, product_embeddings)

    print(f"nearest related products for '{_query_text}'")

    for _hit in _hits:
        _product = products[_hit['index']]
        print(_product['title'])
    
    return


@app.cell
def _(
    create_embeddings,
    create_product_text,
    find_n_closest,
    product_embeddings,
    products,
):
    # recommendation system

    last_product = {'title': 'Building Blocks Deluxe Set',
     'short_description': 'Unleash your creativity with this deluxe set of building blocks for endless fun.',
     'price': 34.99,
     'category': 'Toys',
     'features': ['Includes 500+ colorful building blocks',
      'Promotes STEM learning and creativity',
      'Compatible with other major brick brands',
      'Comes with a durable storage container',
      'Ideal for children ages 3 and up']}

    last_product_text = create_product_text(last_product)
    last_product_embeddings = create_embeddings(last_product_text)[0]

    _hits=find_n_closest(last_product_embeddings, product_embeddings)

    # print(f"nearest related products for '{_query_text}'")

    for _hit in _hits:
        _product = products[_hit['index']]
        print(_product['title'])
    


    return


@app.cell
def _():
    # now account for user viewing history


    user_history=[{'title': 'Remote-Controlled Dinosaur Toy',
      'short_description': 'Roar into action with this remote-controlled dinosaur toy with lifelike movements.',
      'price': 49.99,
      'category': 'Toys',
      'features': ['Realistic dinosaur sound effects',
       'Walks and roars like a real dinosaur',
       'Remote control included',
       'Educational and entertaining']},
     {'title': 'Building Blocks Deluxe Set',
      'short_description': 'Unleash your creativity with this deluxe set of building blocks for endless fun.',
      'price': 34.99,
      'category': 'Toys',
      'features': ['Includes 500+ colorful building blocks',
       'Promotes STEM learning and creativity',
       'Compatible with other major brick brands',
       'Comes with a durable storage container',
       'Ideal for children ages 3 and up']}]
    return (user_history,)


@app.cell
def _(create_embeddings, create_product_text, np, user_history):
    history_texts = [create_product_text(product) for product in user_history]
    history_embeddings = create_embeddings(history_texts)

    # take the mean to create a new vector to compare against
    mean_history_embeddings = np.mean(history_embeddings, axis=0)
    return (mean_history_embeddings,)


@app.cell
def _(
    create_embeddings,
    create_product_text,
    find_n_closest,
    mean_history_embeddings,
    products,
    user_history,
):
    # filter products already looked at in user history
    _products_filtered = [product for product in products if product not in user_history]
    _products_text = [create_product_text(product) for product in _products_filtered]
    _product_embeddings = create_embeddings(_products_text)

    _hits=find_n_closest(mean_history_embeddings,  _product_embeddings)

    # print(f"nearest related products for '{_query_text}'")

    for _hit in _hits:
        _product = _products_filtered[_hit['index']]
        print(_product['title'])
    

    return


@app.cell
def _(products_filtered):
    products_filtered
    return


@app.cell
def _(products):
    products

    return


@app.cell
def _(user_history):
    user_history
    return


@app.cell
def _(
    create_embeddings,
    create_product_text,
    find_n_closest,
    np,
    products,
    user_history,
):
    # Prepare and embed the user_history, and calculate the mean embeddings
    _history_texts = [create_product_text(product) for product in user_history]
    _history_embeddings = create_embeddings(_history_texts)
    _mean_history_embeddings = np.mean(_history_embeddings, axis =0)

    # Filter products to remove any in user_history
    _products_filtered = [_product for _product in products if _product not in user_history]

    # Combine product features and embed the resulting texts
    _product_texts = [create_product_text(_product) for _product in _products_filtered]
    _product_embeddings = create_embeddings(_product_texts)

    _hits = find_n_closest(_mean_history_embeddings, _product_embeddings)

    for _hit in _hits:
      _product = _products_filtered[_hit['index']]
      print(_product['title'])
    return


@app.cell
def _():
    # classification with embeddings
    sentiments = [{'label': 'Positive'}, {'label': 'Neutral'}, {'label': 'Negative'}]
    reviews = ['The food was delicious!',
     'The service was a bit slow but the food was good',
     'The food was cold, really disappointing!']
    return reviews, sentiments


@app.cell
def _(distance):
    def find_closest(query_vector, embeddings):
        distances = []
        for index, embedding in enumerate(embeddings):
            dist = distance.cosine(query_vector, embedding)
            distances.append({"distance": dist, "index": index})
        return min(distances, key=lambda x: x["distance"])

    return (find_closest,)


@app.cell
def _(create_embeddings, find_closest, reviews, sentiments):
    _class_descriptions = [sentiment["label"] for sentiment in sentiments]

    # embed descriptiuons and reviews
    _class_embeddings = create_embeddings(_class_descriptions)
    _review_embeddings = create_embeddings(reviews)

    # print(_review_embeddings)

    for _index, _review in enumerate(reviews):
        _closest = find_closest(_review_embeddings[_index], _class_embeddings)

        label = sentiments[_closest['index']]['label']
        print(f"{_review} was classified as {label}")
    return


@app.cell
def _(reviews):
    for index, review in enumerate(reviews):
        print(index)
        print(review)
    return


@app.cell
def _(create_embeddings, find_closest, reviews, sentiments):
    # the model needed more info to be able to classify neutral
    new_sentiments = [{'label': 'Positive',
                   'description': 'A positive restaurant review'},
                  {'label': 'Neutral',
                   'description':'A neutral restaurant review'},
                  {'label': 'Negative',
                   'description': 'A negative restaurant review'}]

    _class_descriptions = [sentiment["description"] for sentiment in new_sentiments]

    # embed descriptiuons and reviews
    _class_embeddings = create_embeddings(_class_descriptions)
    _review_embeddings = create_embeddings(reviews)

    # print(_review_embeddings)

    for _index, _review in enumerate(reviews):
        _closest = find_closest(_review_embeddings[_index], _class_embeddings)

        _label = sentiments[_closest['index']]['label']
        print(f"{_review} was classified as {_label}")

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
