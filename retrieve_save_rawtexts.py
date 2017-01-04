import os
from paperity.environ import db
import csv
from os.path import exists

def retrieve_rawtexts_by_id(srcfilename, destdir):
    """
    Extract 'rawtext' for all papers which ids are provided in the file srcfilename, each id kept in a separate line,
    and save them to separate files in the provided directory: <destdir>/<pap_id>.txt
    """

    if not exists(destdir):
        raise Exception("No such directory: '" + destdir + "'")        

    paper_ids = []

    f_ids = open(srcfilename, 'r')
    for paper_id in f_ids:
        paper_ids.append(paper_id)

    f_ids.close()

    papers = Paper.select("pid, rawtext", _where = "pid in (%s)" % ', '.join(paper_ids))

    path = destdir + '/%s.txt'
    for paper in papers:
        f_name = path % paper.pid
        f = open(f_name, 'w')
        
        print >>f, paper.rawtext.encode('utf-8')

        f.close()

def save_rawtexts_by_id(rawtext_dirname, paper_list_filename):
    if not exists(rawtext_dirname):
        raise Exception("No such directory: '" + rawtext_dirname + "'")

    if not exists(paper_list_filename):
        raise Exception("No such file: '" + paper_list_filename + "'")       

    with open(paper_list_filename, 'r') as paperfile:
        reader = csv.reader(paperfile, delimiter=',', quotechar='"')
        reader.next() #skipping header
        
        for i, paper_row in enumerate(reader):
            print i
            paper_id = paper_row[3]
            paper_title = paper_row[2].replace("'", "''").decode('utf-8')
            paper_category = paper_row[0] + "," + paper_row[1]
            rawtext_filename = '%s/%s.txt' % (rawtext_dirname, paper_id)
            
            if exists(rawtext_filename):
                rawtext_file = open(rawtext_filename, 'r')
                paper_rawtext = rawtext_file.read().replace("'", "''").decode('utf-8')
                rawtext_file.close()
                db_row_count = db.select("SELECT COUNT(1) FROM pap_papers_view WHERE id = " + str(paper_id))[0]['COUNT(1)']
                
                sql1 = ""
                sql2 = ""
                sql3 = ""

                if db_row_count > 0:
                    sql1 = "update pap_papers_1 set published = 1 where id = %s" % (paper_id, )
                    sql2 = "update pap_papers_2 set title = '%s', learned_category = '%s' where id = %s" % (paper_title, paper_category, paper_id)
                    sql3 = "update pap_papers_3 set rawtext = '%s' where id = %s" % (paper_rawtext, paper_id)
                else:
                    sql1 = "insert into pap_papers_1(id,journal_id,published) values(%s, 1001, 1)" % (paper_id,)
                    sql2 = "insert into pap_papers_2(id,title,learned_category) values(%s, '%s', '%s')" % (paper_id, paper_title, paper_category)
                    sql3 = "insert into pap_papers_3(id,rawtext) values(%s, '%s')" % (paper_id, paper_rawtext)
                #print sql
                
                db.query(sql1)
                db.query(sql2)
                db.query(sql3)
    
    db.commit()

if __name__ == "__main__":
    rawtext_dirname = "/home/cezary/Documents/MGR/rawtexts"
    paper_list_filename = "/home/cezary/Documents/MGR/discipline_allocation.csv"

    save_rawtexts_by_id(rawtext_dirname, paper_list_filename)