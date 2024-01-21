# Submission 3: Exercise Data Mesh
Link to the repository: https://github.com/MaxwellMuF/Submissions_Advanced_Software_Engineering_-WiSe23-/tree/main/Sub_3_Datamesh

## Given website: https://datamesh-manager.com/
Make data management better.
Relatiuon between data owner and data consumer.
Organize data source (data base) and data product (data processing and outcome)

## Description of the demo:
Demo environment of electronically buying or selling (e commerce).
First, the demo shows a data map. This time the processes from the data source to the data product and its customers. The interactive dashboard shows various examples of such a process.

### Click on subprocesses of the data map:
The individual process steps or actors can be clicked on to obtain additional information. 
These functions not only provide details (Info) of the e.g. data product but also offer metrics: Consumers (number), Costs, Compliance (includes policies). If you click further on e.g. costs, the costs of this pipeline are displayed. 
In one case, the software used (Snowflake Compute, Data Cloud) is listed with name, description, category and the costs of this are calculated and later cumulated. The classification of the sub-process in the overall process or pipeline is also specified. The "Input Ports" and "Output Ports" columns describe the position in the data map and the related processes are displayed directly.

### Search option on the data map
A search option can be used to search the entire data map to find specific sub-processes. This is also possible via filters: the "Owner" filter can be set to "Marketing" or "Controlling", for example, in order to filter out the corresponding areas of responsibility. Similarly, a "Status" filter can also be used to filter out active processes in this example.

### Subprocess tab and data map
The "Data Product" tabs offer a similar function "Source Systems" and "Domain Teams" tabs. These again divide the data map into different backup points and organize them as required. The Domain Teams tab, for example, lists the actors: A description of the corresponding teams (e.g. working groups or departments) is available here and the members of this team are listed with their role and a contact option.

Finally, new sub-processes can of course be added under the "Add Data Product" tab. This function opens a clearly structured information and data query for the sub-process to be added. 

### Summary
In summary, datamesh-manager is a system for organizing processes (in this case relating to e commerce) and their individual stations and players. In this way, a wide variety of tasks and areas can be summarized in a map that is very clear. Many details are always available through short clicks and forwarding, down to any depth of the sub-processes and the actors involved. Thanks to the standardized structure, all those involved and interested in the project can use a common workspace without losing the overview due to details of individual areas.
