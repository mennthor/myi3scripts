################################################################################
## This is a mashup between virtualenvs activate script and icetrays env-shell.
## Last edit: 2017, Dec 27th
##
## It prevents all bash settings by storing them in '_I3OLD_*' variables and
## only changes the paths so that icetray can be found. After calling the new
## function `iceout` all paths are reset to the pre-invocation state.
##
## This is meant to be useful for interactive usage because it preserves bash
## aliases the user made, no ne shell is invoked. When used in tray scripts,
## using the standard env-shell.sh should be the way to got.
##
## To use in a project, copy it this script the project build folder next to the
## original env-shell.sh and use 'source ./activate.sh' to load the environment.
## Or set an alias in bashrc:
##   alias icecombo='source /path/to/combo/build/activate.sh'
##
## Per default the PS1 is changed to include the dirname of the project. If
## this is not wanted, invoke with
##   source /path/to/combo/build/activate.sh --NOPS1
################################################################################

# Check script args
if [ "--NOPS1" == "$1" ] ; then
    _NOPS1=TRUE
fi

iceout () {
    ##########################################################################
    ## Function to exit the environment by resetting the complete old state
    ##########################################################################
    # If old vars were empty initially, make them empty again
    if [ "$_I3OLD_PATH" == "_ICEEMPTY_" ] ; then
        export PATH="";
        unset _I3OLD_PATH
    fi
    if [ "$_I3OLD_LD_LIBRARY_PATH" == "_ICEEMPTY_" ] ; then
        export LD_LIBRARY_PATH="";
        unset _I3OLD_LD_LIBRARY_PATH
    fi
    if [ "$_I3OLD_DYLD_LIBRARY_PATH" == "_ICEEMPTY_" ] ; then
        export DYLD_LIBRARY_PATH="";
        unset _I3OLD_DYLD_LIBRARY_PATH
    fi
    if [ "$_I3OLD_PYTHONPATH" == "_ICEEMPTY_" ] ; then
        export PYTHONPATH="";
        unset _I3OLD_PYTHONPATH
    fi

    # Reset environment variables if old ones were valid
    if [ -n "$_I3OLD_PATH" ] ; then
        export PATH="$_I3OLD_PATH"
        unset _I3OLD_PATH
    fi
    if [ -n "$_I3OLD_LD_LIBRARY_PATH" ] ; then
        export LD_LIBRARY_PATH="$_I3OLD_LD_LIBRARY_PATH"
        unset _I3OLD_LD_LIBRARY_PATH
    fi
    if [ -n "$_I3OLD_DYLD_LIBRARY_PATH" ] ; then
        export DYLD_LIBRARY_PATH="$_I3OLD_DYLD_LIBRARY_PATH"
        unset _I3OLD_DYLD_LIBRARY_PATH
    fi
    if [ -n "$_I3OLD_PYTHONPATH" ] ; then
        export PYTHONPATH="$_I3OLD_PYTHONPATH"
        unset _I3OLD_PYTHONPATH
    fi

    # This should detect bash and zsh, which have a hash command that must
    # be called to get it to forget past commands.  Without forgetting
    # past commands the $PATH changes we made may not be respected
    if [ -n "$BASH" -o -n "$ZSH_VERSION" ] ; then
        hash -r 2>/dev/null
    fi

    # Reset old PS1 if a new one was set
    if [ -n "$_I3OLD_PS1" ] ; then
        PS1="$_I3OLD_PS1"
        export PS1
        unset _I3OLD_PS1
    fi

    # Unset all created I3 environment variables
    unset I3_SHELL
    unset I3_SRC
    unset I3_BUILD
    unset I3_TESTDATA
    if [ ! "$1" = "nondestructive" ] ; then
    # Self destruct!
        unset -f iceout
    fi
}

##############################################################################
## Activate like virtualenv with env-shell paths
##############################################################################
# First test if we are in a activated shell
if [ -z "$I3_SHELL" ]  ; then
    # First unset possible leftover variables, fresh start
    iceout nondestructive

    # Get script location. From https://stackoverflow.com/questions/59895
    _DIRNAME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    export I3_SHELL="$( cd "$( dirname "$_DIRNAME" )" && pwd )"

    # Set project paths
    _I3_SRC=${I3_SHELL}/src
    _I3_BUILD=${I3_SHELL}/build

    # Check if I3_TESTDATA is given externally and valid
    _I3_TESTDATA=
    if [ -d "$I3_TESTDATA" ] ; then
        if [ $(readlink "$_I3_TESTDATA") != $(readlink "$I3_TESTDATA") ]
        then
            _I3_TESTDATA=$I3_TESTDATA
        fi
    fi

    # Get the SVN revision this project was (probably) compiled against
    # (wrong if SVN was updated but not recompiled yet...)
    _REVISION="$( cd "$_I3_SRC" && svn info | grep Revision | awk '{print $2}' )"

    # Get python version
    _PYVER=`python -V 2>&1`

    # Check for ROOT and set paths accordingly if it exists. If not, leave empty
    if [ -d "$ROOTSYS" ] ; then
    _ROOTSYS=$ROOTSYS
    _PATH=${I3_SHELL}/build/bin:${_ROOTSYS}/bin:$_I3_PORTS/bin:$PATH
    _LD_LIBRARY_PATH=${I3_SHELL}/build/lib:${I3_SHELL}/build/lib/tools:${_ROOTSYS}/lib:$LD_LIBRARY_PATH
    _DYLD_LIBRARY_PATH=${I3_SHELL}/build/lib:${I3_SHELL}/build/lib/tools:${_ROOTSYS}/lib:$_I3_PORTS/lib:$DYLD_LIBRARY_PATH
    _PYTHONPATH=${I3_SHELL}/build/lib:${_ROOTSYS}/lib:$PYTHONPATH
    else
    _ROOTSYS=
    _PATH=${I3_SHELL}/build/bin:$_I3_PORTS/bin:$PATH
    _LD_LIBRARY_PATH=${I3_SHELL}/build/lib:${I3_SHELL}/build/lib/tools::$LD_LIBRARY_PATH
    _DYLD_LIBRARY_PATH=${I3_SHELL}/build/lib:${I3_SHELL}/build/lib/tools:$_I3_PORTS/lib:$DYLD_LIBRARY_PATH
    _PYTHONPATH=${I3_SHELL}/build/lib:$PYTHONPATH
    fi

    # Save the old state before assigning new variables
    _I3OLD_PATH="$PATH"
    _I3OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
    _I3OLD_DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH"
    _I3OLD_PYTHONPATH="$PYTHONPATH"

    # Hack to set empty strings to a value, otherwise the ifs in deactivate
    # don't reset the variables if they were initially empty
    if [ -z "$_I3OLD_PATH" ] ; then _I3OLD_PATH="_ICEEMPTY_"; fi
    if [ -z "$_I3OLD_LD_LIBRARY_PATH" ] ; then _I3OLD_LD_LIBRARY_PATH="_ICEEMPTY_"; fi
    if [ -z "$_I3OLD_DYLD_LIBRARY_PATH" ] ; then _I3OLD_DYLD_LIBRARY_PATH="_ICEEMPTY_"; fi
    if [ -z "$_I3OLD_PYTHONPATH" ] ; then _I3OLD_PYTHONPATH="_ICEEMPTY_"; fi

    # Set all updated enviroment variables to activate the environment
    export PATH=$_PATH
    export LD_LIBRARY_PATH=$_LD_LIBRARY_PATH
    export DYLD_LIBRARY_PATH=$_DYLD_LIBRARY_PATH
    export PYTHONPATH=$_PYTHONPATH
    # And set all new variables. Those are unset on deactivate
    export I3_SRC=$_I3_SRC
    export I3_BUILD=$_I3_BUILD
    export I3_TESTDATA=$_I3_TESTDATA

    # Set new PS1 to project basename to indicate we are in a env-shell
    # if [ -z "$I3_SHELL_DISABLE_PROMPT" ] ; then
    if [ -z $_NOPS1 ] ; then
        _I3OLD_PS1="$PS1"
        if [ "x" != x ] ; then
            PS1="$PS1"
        else
        if [ "`basename \"$I3_SHELL\"`" = "__" ] ; then
            # special case for Aspen magic directories
            # see http://www.zetadev.com/software/aspen/
            PS1="[`basename \`dirname \"$I3_SHELL\"\``] $PS1"
        else
            # PS1="(`basename \"$I3_SHELL\"`)$PS1"
            PS1="\[\e[33m\] (`basename \"$I3_SHELL\"`)\[\e[m\] $PS1"
        fi
        fi
        export PS1
    fi

    # This should detect bash and zsh, which have a hash command that must
    # be called to get it to forget past commands.  Without forgetting
    # past commands the $PATH changes we made may not be respected
    if [ -n "$BASH" -o -n "$ZSH_VERSION" ] ; then
        hash -r 2>/dev/null
    fi

    # Print activation message
    TOPBAR="************************************************************************"
    WIDTH=`echo "$TOPBAR" | wc -c`
    WIDTH=$(( $WIDTH-2 ))
    printctr()
    {
        LEN=`echo "$*" | wc -c`
        LOFFSET=$(( ($WIDTH-$LEN)/2 ))
        ROFFSET=$(( $WIDTH-$LOFFSET-$LEN ))
        FORMAT="*%${LOFFSET}s%s%${ROFFSET}s*\n"
        printf $FORMAT " " "$*" " "
    }
    if [ -z "$ARGV" ] ; then
        printf "$TOPBAR\n"
        printctr ""
        printctr "W E L C O M E  to  I C E T R A Y"
        printctr ""
        printctr "Version combo.trunk      r$_REVISION"
        printctr ""
        printf "$TOPBAR\n"
        # printf "\n"
        printf "Icetray environment has:\n"
        printf "   I3_SRC       = %s\n" $_I3_SRC
        printf "   I3_BUILD     = %s\n" $_I3_BUILD
        [ -z "$_I3_PORTS" ] || printf "   I3_PORTS     = %s\n" $_I3_PORTS
        [ -d "$_I3_TESTDATA" ] && printf "   I3_TESTDATA  = %s\n" $_I3_TESTDATA \
                               || printf "   I3_TESTDATA  = Should be set to an existing directory path\n" \
                                         "   (and 'make rsync' may need to be run) if you wish to run tests."
        echo   "   Python       =" $_PYVER
        # Check if meta-project was built with -DUSE_ROOT=TRUE
        _META_HOME_VAR=$(grep -F 'METAPROJECT:STRING' $_I3_BUILD/CMakeCache.txt)
        _META_HOME_PATH=${_META_HOME_VAR#'METAPROJECT:STRING='}
        if [ ! -d "$_META_HOME_PATH" ] ; then _META_HOME_PATH=$_I3_BUILD ; fi
        if grep -Fxq "USE_ROOT:BOOL=TRUE" $_META_HOME_PATH/CMakeCache.txt ; then
            printf "   ROOT         = %s\n" $_ROOTSYS
        else
            printf "   ROOT         = Meta-project not build with ROOT.\n"
        fi
        printf "When finished, exit environment calling 'iceout'.\n"
    fi

    # Clean up and unset leftover temp vars used above
    unset _I3_BUILD
    unset _I3_PORTS
    unset _I3_SRC
    unset _I3_TESTDATA

    unset _PATH
    unset _PYVER
    unset _PYTHONPATH
    unset _LD_LIBRARY_PATH
    unset _DYLD_LIBRARY_PATH
    unset _ROOTSYS
    unset _REVISION

    unset _DIRNAME
    unset _NOPS1
    unset _META_HOME_VAR
    unset _META_HOME_PATH

else
    # If an I3 environment is already active, do nothing
    echo "We already are in an I3 environment: $I3_SHELL"
    echo "Nothing was changed. You may exit with 'iceout'."
fi