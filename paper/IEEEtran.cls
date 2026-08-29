% IEEEtran.cls style definitions for IEEE conference papers
% Standard minimal self-contained IEEE conference document class for compilation
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{IEEEtran}[2026/01/01 V1.8 by IEEE]
\LoadClass[10pt,conference,letterpaper,twocolumn]{article}

\RequirePackage{geometry}
\geometry{letterpaper,top=0.75in,bottom=1.0in,left=0.625in,right=0.625in,columnsep=0.25in}

\RequirePackage{cite}
\RequirePackage{amsmath,amssymb,amsfonts}
\RequirePackage{algorithmic}
\RequirePackage{graphicx}
\RequirePackage{textcomp}
\RequirePackage{xcolor}
\RequirePackage{booktabs}
\RequirePackage{microtype}
\RequirePackage{hyperref}

\hypersetup{
    colorlinks=true,
    linkcolor=black,
    citecolor=black,
    urlcolor=blue
}

\pagestyle{empty}
