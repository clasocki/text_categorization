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

def compare_category_allocation(filename1, filename2, filename_same, filename_diff):
    allocations1, allocations2 = dict(), dict()
    with open(filename1, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"', lineterminator='\n')

        for row in reader:
            if len(row) == 5:
                pap_id = row[0]
                title = row[2]
                category = row[3]
                allocations1[title] = category

    num_invalid_titles = 0
    with open(filename2, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"', lineterminator='\n')        
        for row in reader:
            if len(row) == 4:
                pap_id = row[0]
                title = row[2]
                category = row[3]
                allocations2[title] = category
            else:
                num_invalid_titles += 1

    matches = 0
    papers_in_both_sets = 0

    with open(filename_same, 'w') as f_same, open(filename_diff, 'w') as f_diff:
        same_csv_writer = csv.writer(f_same)
        diff_csv_writer = csv.writer(f_diff)
        for title, category in allocations2.iteritems():
            print title, category
            if title in allocations1:
                papers_in_both_sets += 1
                if allocations1[title] == category:
                    matches += 1
                    same_csv_writer.writerow([title, category])
                else:
                    diff_csv_writer.writerow([title, category, allocations1[title]])

    print "Matching categories: %s out of %s, which accounts for %s" % (matches, papers_in_both_sets, float(matches) / papers_in_both_sets)
    print "Invalid: %s" % (num_invalid_titles, )

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
            #paper_category = paper_row[0] + "," + paper_row[1]
            paper_category = paper_row[0]
            rawtext_filename = '%s/%s.txt' % (rawtext_dirname, paper_id)
            
            if exists(rawtext_filename):
                rawtext_file = open(rawtext_filename, 'r')
                paper_rawtext = rawtext_file.read().replace("'", "''").decode('utf-8')
                rawtext_file.close()
                db_row_count = db.select("SELECT COUNT(1) FROM pap_papers_view WHERE pid = " + str(paper_id))[0]['COUNT(1)']
                
                sql1 = ""
                sql2 = ""
                sql3 = ""

                if db_row_count > 0:
                    sql1 = "update pap_papers_1 set published = 1 where pid = %s" % (paper_id, )
                    sql2 = "update pap_papers_2 p2 left outer join pap_papers_1 p1 on p1.id = p2.id "
                    sql2 += "set p2.title = '%s', p2.learned_category = '%s' where p1.pid = %s" % (paper_title, paper_category, paper_id)
                    sql3 = "update pap_papers_3 p3 left outer join pap_papers_1 p1 on p1.id = p3.id "
                    sql3 += "set p3.rawtext = '%s' where p1.pid = %s" % (paper_rawtext, paper_id)
                else:
                    new_id = int(db.select("select max(id) from pap_papers_1")[0]['max(id)']) + 1
                    sql1 = "insert into pap_papers_1(id, pid,journal_id,published) values(%s, %s, 1001, 1)" % (new_id, paper_id,)
                    sql2 = "insert into pap_papers_2(id, title,learned_category) values(%s, '%s', '%s')" % (new_id, paper_title, paper_category)
                    sql3 = "insert into pap_papers_3(id, rawtext) values(%s, '%s')" % (new_id, paper_rawtext)
                #print sql
                
                db.query(sql1)
                db.query(sql2)
                db.query(sql3)
    
    db.commit()

if __name__ == "__main__":
    rawtext_dirname = "/home/clasocki/rawtexts"
    paper_list_filename = "/home/clasocki/discipline_allocation.csv"

    save_rawtexts_by_id(rawtext_dirname, paper_list_filename)

    dirname = '/home/cezary/Documents/MGR/backups/'
    #compare_category_allocation(dirname + 'labeled_iterative.csv', dirname + 'labeled_gensim.csv', dirname + 'same_categories.csv', dirname + 'diff_categories.csv')
