# -*- coding: utf-8 -*-  GetOldTweets-python
import sys,getopt,datetime,codecs
from collections import Counter
if sys.version_info[0] < 3:
    import got
else:
    import got3 as got

def getTweets(name,query,since,until):
	tweetCriteria = got.manager.TweetCriteria()
	outputFileName = name
	#tweetCriteria.username = arg
	tweetCriteria.since = since
	tweetCriteria.until = until
	tweetCriteria.querySearch = query
	tweetCriteria.topTweets = True
	tweetCriteria.maxTweets = 10000
	#tweetCriteria.near = '"' + arg + '"'
	#tweetCriteria.within = '"' + arg + '"'			
	outputFile = codecs.open(outputFileName, "w+", "utf-8")
	outputFile.write('username;date;retweets;favorites;text;geo;mentions;hashtags;id;permalink')

	print('Searching...\n')

	def receiveBuffer(tweets):
		for t in tweets:
			outputFile.write(('\n%s;%s;%d;%d;"%s";%s;%s;%s;"%s";%s' % (t.username, t.date.strftime("%Y-%m-%d %H:%M"), t.retweets, t.favorites, t.text, t.geo, t.mentions, t.hashtags, t.id, t.permalink)))
		outputFile.flush()
		print('More %d saved on file...\n' % len(tweets))

	got.manager.TweetManager.getTweets(tweetCriteria, receiveBuffer)
	#tweetCriteria = got.manager.TweetCriteria().setQuerySearch('europe refugees').setSince("2015-05-01").setUntil("2015-09-30").setMaxTweets(1)

	outputFile.close()
	print('Done. Output file generated "%s".' % outputFileName)
	return outputFileName

def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts

#print(Counter(open("output_got.csv","r+", encoding="utf8").read()).most_common(100))

'''
# -*- coding: utf-8 -*-
import sys,getopt,datetime,codecs
if sys.version_info[0] < 3:
    import got
else:
    import got3 as got

def main(argv):

	if len(argv) == 0:
		print('You must pass some parameters. Use \"-h\" to help.')
		return

	if len(argv) == 1 and argv[0] == '-h':
		f = open('exporter_help_text.txt', 'r')
		print(f.read())
		f.close()

		return

	try:
		opts, args = getopt.getopt(argv, "", ("username=", "near=", "within=", "since=", "until=", "querysearch=", "toptweets", "maxtweets=", "output="))

		tweetCriteria = got.manager.TweetCriteria()
		outputFileName = "output_got.csv"

		for opt,arg in opts:
			if opt == '--username':
				tweetCriteria.username = arg

			elif opt == '--since':
				tweetCriteria.since = arg

			elif opt == '--until':
				tweetCriteria.until = arg

			elif opt == '--querysearch':
				tweetCriteria.querySearch = arg

			elif opt == '--toptweets':
				tweetCriteria.topTweets = True

			elif opt == '--maxtweets':
				tweetCriteria.maxTweets = int(arg)
			
			elif opt == '--near':
				tweetCriteria.near = '"' + arg + '"'
			
			elif opt == '--within':
				tweetCriteria.within = '"' + arg + '"'

			elif opt == '--within':
				tweetCriteria.within = '"' + arg + '"'

			elif opt == '--output':
				outputFileName = arg
				
		outputFile = codecs.open(outputFileName, "w+", "utf-8")

		outputFile.write('username;date;retweets;favorites;text;geo;mentions;hashtags;id;permalink')

		print('Searching...\n')

		def receiveBuffer(tweets):
			for t in tweets:
				outputFile.write(('\n%s;%s;%d;%d;"%s";%s;%s;%s;"%s";%s' % (t.username, t.date.strftime("%Y-%m-%d %H:%M"), t.retweets, t.favorites, t.text, t.geo, t.mentions, t.hashtags, t.id, t.permalink)))
			outputFile.flush()
			print('More %d saved on file...\n' % len(tweets))

		got.manager.TweetManager.getTweets(tweetCriteria, receiveBuffer)

	except arg:
		print('Arguments parser error, try -h' + arg)
	finally:
		outputFile.close()
		print('Done. Output file generated "%s".' % outputFileName)

if __name__ == '__main__':
	main(sys.argv[1:])
'''