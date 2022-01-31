# The Canonizer

Installation:

	$ git clone https://github.com/KBNLresearch/DBNL-canonicity.git
	$ cd DBNL-canonicity/demo
	$ wget https://urd2.let.rug.nl/~andreas/canonizer/data.zip
	$ unzip data.zip
    $ rm data.zip

Run:

	$ pip3 install -r requirements.txt
	$ gunicorn --bind=0.0.0.0:5004 --workers=4 --preload web:APP

Run with Docker, automatically restarts at reboot:

	$ docker build -t canonizer .
	$ docker run -p 5004:5004 --restart always canonizer
