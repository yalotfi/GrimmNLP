import os

from bs4 import BeautifulSoup


def parse_story(page):
    """Take a filepath to HTML file to be parsed into story"""
    soup = BeautifulSoup(open(page), 'lxml')
    title = soup.find('h1').text.strip()
    body = soup.find('body')
    story_list = []
    for element in body.p.next_siblings:
        if element.name is 'p':
            story_list.append(element.text.strip().replace('\n', ' '))
    story = '\n'.join(story_list[:-1])
    return title, story


def write_corpus(titles, stories, outfile):
    """Write a list of titles and stories to disk"""
    print("Saving stories in {}\n".format(outfile))
    with open(outfile, "w") as ftxt:
        for title, story in zip(titles, stories):
            print("Writing {}...".format(title))
            ftxt.write("\n\n{0}\n\n{1}".format(title, story))
    print("\nFinishined writing the Grimm Corpus...")


def process_grimm(outfile):
    grimm_files = os.listdir('data/')
    n_stories = len(grimm_files)
    grimm_titles = []
    grimm_stories = []
    story_counter = 0
    for file in range(n_stories):
        page = os.path.join('data', grimm_files[file])
        if os.stat(page).st_size > 2000:
            story_counter += 1
            title, story = parse_story(page)
            grimm_titles.append(title)
            grimm_stories.append(story)
    write_corpus(grimm_titles, grimm_stories, outfile)
    print("\nParsed {} stories".format(story_counter))


if __name__ == '__main__':
    outfile = "grimm_corpus.txt"
    process_grimm(outfile)
