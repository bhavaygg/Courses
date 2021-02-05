import subprocess
name= "SRR390728"
bashCommand = "fastq-dump --split-3 "+name
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

s_p=0
ref_genome = "chr1.fa" #aligning to chromosome 1 bause whole genome was giving memory issues on my machine
if s_p==0: #for single end
    bashCommand = "fastqc "+name+".fastq"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()


    bashCommand = "bowtie2-build "+ref_genome+" index/abc" #create folder named index before this
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
    bashCommand = "bowtie2 -x index/abc -U "+name+".fastqc -s abc.sam" 
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
    bashCommand = "samtools view -S -b abc.sam > abc.bam" 
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    bashCommand = "samtools sort abc.bam -o sabc.bam" 
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    bashCommand = "samtools index sabc.bam" 
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

else:#for pair ended
    bashCommand = "fastqc "+name+"_1.fastq"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    bashCommand = "fastqc "+name+"_2.fastq"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    bashCommand = "bowtie2-build "+ref_genome+" index/abc" #create folder named index before this
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
    bashCommand = "bowtie2 -x index/abc -1 "+name+"_1.fastqc -2 "+name+"_2.fastqc -s abc.sam" 
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
    bashCommand = "samtools view -S -b abc.sam > abc.bam" 
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    bashCommand = "samtools sort abc.bam -o sabc.bam" 
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    bashCommand = "samtools index sabc.bam" 
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()


