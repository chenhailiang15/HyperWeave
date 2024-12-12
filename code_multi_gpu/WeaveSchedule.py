




class WeaveSchedulor:
    def __init__(self, strategy):
        self.strategy=strategy
    
    def do_schedule(self,job_list):
        if self.strategy=="random":
            rest_job=self.schedule_random(job_list)
        else:
            print("strategy wrong!")
            exit(-1)
        return rest_job
    
    def schedule_random(self, job_list):
        for job in job_list:
            print("execute: ",job)
        
        
        return []