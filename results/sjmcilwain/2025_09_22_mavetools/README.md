need to install postgres for mavetools
Some issue with tables?

I recommend that you try to use Postgres.app. (http://postgresapp.com) This way you can easily turn Postgres on and off on your Mac. Once you do, add the path to Postgres to your .profile file by appending the following:

PATH="/Applications/Postgres.app/Contents/Versions/latest/bin:$PATH"
Only after you added Postgres to your path you can try to install psycopg2 either within a virtual environment (using pip) or into your global site packages.
	

1006  conda create --name mavedb python=3.8 -y
 1007  conda activate mavedb
 1008  pip3 install mavetools
 1009  pip3 install jupyter

I ended up punting on the mavetools and just directly accessing the api using curl and jsonlite in R.

