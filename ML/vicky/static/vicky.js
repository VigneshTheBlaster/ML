window.onload=function(){
        var availability_of_tickets=(9>localStorage["cont"])?9-localStorage["cont"]:9;
        document.getElementById("td").addEventListener("click",()=>{
                if(localStorage["check"]==1){
                if(localStorage["cont"]<=9){
                document.getElementsByTagName("div")[1].style.display="none";
                document.getElementsByTagName("div")[0].style.display="none";
                document.getElementsByTagName("table")[0].style.display="table";
                var frm= localStorage["frm"];
                var to= localStorage["to"];
                var trn= localStorage["trn"];
                var coh= localStorage["coh"];
                var Date= localStorage["date"];
                var tm= localStorage["tm"];
                var mg= localStorage["mg"];
                var cont= localStorage["cont"];
                document.getElementById("fm").innerText=frm;
                document.getElementById("ot").innerText=to;
                document.getElementById("tn").innerText=trn;
                document.getElementById("nc").innerText=coh;
                document.getElementById("ad").innerText=Date;
                document.getElementById("mt").innerText=tm;
                document.getElementById("eg").innerText=mg;
                document.getElementById("uc").innerText=cont;
                }
                else{
                        localStorage["check"]=0;
                        document.getElementsByTagName("div")[1].style.display="none";
                        document.getElementsByTagName("div")[0].style.display="block";
                        document.getElementsByTagName("div")[0].style.marginTop="200px";
                        document.getElementsByTagName("div")[0].innerHTML="<h1>Tickets not available</h1>";
                }
                }
                else{
                        alert("No Form Is Not Filled");
                        localStorage["check"]=0;
                }
        });
        document.getElementById("dt").addEventListener("click",()=>{
                document.getElementsByTagName("table")[0].style.display="none";
                document.getElementsByTagName("div")[0].style.display="none";
                document.getElementsByTagName("div")[1].style.display="block";
                document.getElementsByTagName("div")[1].style.marginTop="200px";
                document.getElementsByTagName("div")[1].innerHTML="<h1>Only "+availability_of_tickets+" Tickects Available Now</h1>";
        });
}