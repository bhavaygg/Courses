import requests
target="SRR001666"
link="ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR494/SRR494099/SRR494099.fastq.gz"
r = requests.get(link, stream=True)
with open("test.fasta", 'wb') as fd:
    for chunk in r.iter_content(chunk_size=128):
        print(chunk)
        fd.write(chunk)