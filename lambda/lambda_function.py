import io
import os
import sys
import json
import string

import boto3
import numpy as np
from hashlib import md5
from botocore.exceptions import ClientError
from sagemaker.mxnet.model import MXNetPredictor


if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans
    
def vectorize_sequences(sequences, vocabulary_length):
    results = np.zeros((len(sequences), vocabulary_length))
    for i, sequence in enumerate(sequences):
       results[i, sequence] = 1. 
    return results

def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).
    # Arguments
        text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]

def one_hot(text, n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
    """One-hot encodes a text into a list of word indexes of size n.
    This is a wrapper to the `hashing_trick` function using `hash` as the
    hashing function; unicity of word to index mapping non-guaranteed.
    # Arguments
        text: Input text (string).
        n: int. Size of vocabulary.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        List of integers in [1, n]. Each integer encodes a word
        (unicity non-guaranteed).
    """
    return hashing_trick(text, n,
                         hash_function='md5',
                         filters=filters,
                         lower=lower,
                         split=split)


def hashing_trick(text, n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):
    """Converts a text to a sequence of indexes in a fixed-size hashing space.
    # Arguments
        text: Input text (string).
        n: Dimension of the hashing space.
        hash_function: defaults to python `hash` function, can be 'md5' or
            any function that takes in input a string and returns a int.
            Note that 'hash' is not a stable hashing function, so
            it is not consistent across different runs, while 'md5'
            is a stable hashing function.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of integer word indices (unicity non-guaranteed).
    `0` is a reserved index that won't be assigned to any word.
    Two or more words may be assigned to the same index, due to possible
    collisions by the hashing function.
    The [probability](
        https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)
    of a collision is in relation to the dimension of the hashing space and
    the number of distinct objects.
    """
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]

class Email:
    def __init__(self, sender_email, sender_name, date, 
                 receipient_email, subject, body):
        self.sender_email = sender_email
        self.sender_name = sender_name
        self.date = date
        self.receipient_email = receipient_email
        self.subject = subject
        self.body = body
        
def parse_email(email_raw):
    sender = email_raw.split("From: ")[1].split("\n")[0]
    sender_email = sender.split("<")[1].split(">")[0]
    sender_name = sender.split("<")[0]
    if len(sender_name):
        sender_name = sender_name[:-1]
    
    receipient_email = ""
    if "In-Reply-To: " in email_raw:
        receipient_email = email_raw.split("To: ")[2].split("\r\n")[0]
    else:
        receipient_email = email_raw.split("To: ")[1].split("\r\n")[0]
    
    subject = email_raw.split("Subject: ")[1].split("\r\n")[0]
    
    date = email_raw.split(
        "for <{}>; ".format(receipient_email)
    )[1].split("\r\n")[0]
    
    body = email_raw.split(
        'Content-Type: text/plain; charset="UTF-8"\r\n\r\n'
    )[1].split("\r\n\r\n--")[0]
    
    body = body.replace("\n", " ")
    body = body.replace("\r", " ")
    
    return Email(sender_email, sender_name, date,
                 receipient_email, subject, body)
    
def send_email(email, label, confidence):
    LABELS = ["HAM", "SPAM"]
    # Replace sender@example.com with your "From" address.
    # This address must be verified with Amazon SES.
    SENDER = "AWS Bot <{}>".format(email.receipient_email)
    
    # Replace recipient@example.com with a "To" address. If your account 
    # is still in the sandbox, this address must be verified.
    RECIPIENT = email.sender_email
    
    # Specify a configuration set. If you do not want to use a configuration
    # set, comment the following variable, and the 
    # ConfigurationSetName=CONFIGURATION_SET argument below.
    # CONFIGURATION_SET = "ConfigSet"
    
    # If necessary, replace us-west-2 with the AWS Region you're using for Amazon SES.
    AWS_REGION = "us-east-1"
    
    # The subject line for the email.
    SUBJECT = "Response of [{}]".format(email.subject)
    
    # The email body for recipients with non-HTML email clients.
    BODY_TEXT = (
        "We received your email sent at {} "
        "with the subject {}.\n\n"
        "Here is a 240 character sample of the email body: \n"
        "{}\n\n"
        "The email was categorized as {} with a "
        "{}% confidence."
    ).format(
        email.date,
        email.subject,
        email.body[:min(240,len(email.body))],
        LABELS[int(label)],
        int(confidence*100)
    )
                
    # The HTML body of the email.
    BODY_HTML = """<html>
    <head></head>
    <body>
      <p>{}</p>
    </body>
    </html>
    """.format(BODY_TEXT)            
    
    # The character encoding for the email.
    CHARSET = "UTF-8"
    
    # Create a new SES resource and specify a region.
    client = boto3.client('ses', region_name=AWS_REGION)
    
    # Try to send the email.
    try:
        #Provide the contents of the email.
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {
                    'Html': {
                        'Charset': CHARSET,
                        'Data': BODY_HTML,
                    },
                    'Text': {
                        'Charset': CHARSET,
                        'Data': BODY_TEXT,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=SENDER,
            # If you are not using a configuration set, comment or delete the
            # following line
            # ConfigurationSetName=CONFIGURATION_SET,
        )
        print(response)
    # Display an error if something goes wrong.	
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])
        
        
def lambda_handler(event, context):
    # TODO implement
    s3_client = boto3.client('s3')
    bucket = event['Records'][0]['s3']['bucket']['name']
    name = event['Records'][0]['s3']['object']['key']
    
    s3_object = s3_client.get_object(
        Bucket=bucket,
        Key=name,
    )
    email_raw = io.BytesIO(s3_object['Body'].read())
    
    email = parse_email(email_raw.getvalue().decode("utf-8"))
    print(email.sender_email, email.sender_name, email.receipient_email,
          email.subject, email.body)
    
    # Uncomment the following line to connect to an existing endpoint.
    vocabulary_length = 9013
    endpoint_name = os.environ['ENDPOINT_NAME']
    mxnet_pred = MXNetPredictor(endpoint_name)
    
    test_messages = [email.body]
    one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
    
    result = mxnet_pred.predict(encoded_test_messages)
    label = result['predicted_label'][0][0]
    confidence = result['predicted_probability'][0][0]
    print(label, confidence)
    
    send_email(email, label, confidence)
    
    return {
        'statusCode': 200,
        'body': json.dumps('Message is successfully sent')
    }
