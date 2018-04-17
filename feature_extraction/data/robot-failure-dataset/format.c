      #include <stdio.h>

      char str[128];

      main(int argc,char **argv)
      {
      
         FILE *f1, *f2;
         int i,j,Nex;
         int aux;
      
         f1 = fopen(argv[1],"r");
         f2 = fopen(argv[2],"w");
      
         fscanf(f1,"%d",&Nex);
      
         for(i=0; i<Nex; i++) {
           fscanf(f1,"%s",&str[0]);
           for(j=0; j<90; j++) {
             fscanf(f1,"%d",&aux);
             fprintf(f2,"%d,",aux);
           }
           fprintf(f2,"%s\n",str);                     
         }
      }
